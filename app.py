"""
Book Haven - Book Scraping and Recommendation System
Author: Hadi Assi
"""
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

def scrape_books_to_scrape(pages=2):
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"
    books = []
    for page in range(1, pages + 1):
        res = requests.get(base_url.format(page), headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if res.status_code != 200:
            continue
        soup = BeautifulSoup(res.text, 'html.parser')
        for art in soup.select('article.product_pod'):
            title = art.h3.a['title']
            price_text = art.select_one('p.price_color').text.strip()
            cleaned = re.sub(r'[^0-9.]', '', price_text)
            price = float(cleaned) if cleaned else np.nan
            rating_class = art.select_one('p.star-rating')['class'][1]
            rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
            rating = rating_map.get(rating_class, np.nan)
            image_url = "http://books.toscrape.com/" + art.find("img")['src'].replace('../', '')
            books.append({'title': title, 'price': price, 'rating': rating, 'source': 'BooksToScrape', 'image': image_url})
    return books

def recommend_books(books, top_n=5):
    titles = [book['title'] for book in books]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(titles)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    recommendations = []
    for idx, title in enumerate(titles):
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        recs = [titles[i[0]] for i in sim_scores]
        recommendations.append({'title': title, 'recs': recs})
    return recommendations

def analyze_sentiments(books):
    ratings = pd.Series([b['rating'] for b in books if not pd.isna(b['rating'])])
    return {
        'average_rating': round(ratings.mean(), 2),
        'highest_rating': ratings.max(),
        'lowest_rating': ratings.min(),
        'count': len(ratings),
        'rating_distribution': ratings.value_counts().sort_index().to_dict()
    }

@app.route("/")
def index():
    books = scrape_books_to_scrape(2)
    stats = analyze_sentiments(books) if books else {}
    recommendations = recommend_books(books) if books else []
    title_to_recs = {rec['title']: rec['recs'] for rec in recommendations} if recommendations else {}
    chart_labels = json.dumps(list(stats['rating_distribution'].keys())) if stats else json.dumps([])
    chart_values = json.dumps(list(stats['rating_distribution'].values())) if stats else json.dumps([])

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Book Haven</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: #fff;
            }
            nav {
                background-color: #2d2d2d;
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            }
            nav a {
                color: #fff;
                margin: 0 25px;
                text-decoration: none;
                font-weight: bold;
                font-size: 18px;
                transition: color 0.3s;
            }
            nav a:hover {
                color: #ffd700;
            }
            .book-card {
                background-color: #fff;
                color: #333;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                padding: 15px;
                margin: 15px;
                width: 220px;
                transition: transform 0.3s;
            }
            .book-card:hover {
                transform: translateY(-10px);
            }
            .book-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                padding: 30px 10px;
            }
            h1, h2 {
                text-align: center;
                margin-top: 30px;
            }
            canvas {
                display: block;
                margin: 40px auto;
            }
        </style>
    </head>
    <body>
    <nav>
        <a href="/">🏠 Home</a>
        <a href="/order">🛒 Order</a>
    </nav>
    <h1>📚 Explore Our Curated Book Collection</h1>
    <div class="book-container">
    {% for book in books %}
        <div class="book-card">
            <img src='{{ book.image }}' style='width:100%; height:150px; object-fit: cover; border-radius: 5px;'><br>
            <strong>{{ book.title }}</strong><br>
            <small>Price: ${{ book.price }} | Rating: {{ book.rating }}</small><br>
            <em>Recommendations:</em>
            <ul>
            {% for rec in title_to_recs.get(book.title, []) %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ul>
        </div>
    {% endfor %}
    </div>

    <h2>📊 Book Rating Overview</h2>
    <canvas id="ratingChart" width="400" height="200"></canvas>
    <script>
      const data = {
        labels: {{ chart_labels | safe }},
        datasets: [{
          label: 'Rating Distribution',
          data: {{ chart_values | safe }},
          backgroundColor: ['#ff595e', '#ffca3a', '#8ac926', '#1982c4', '#6a4c93']
        }]
      };
      new Chart(document.getElementById('ratingChart'), {
        type: 'bar',
        data: data
      });
    </script>
    </body>
    </html>
    """, books=books, stats=stats, title_to_recs=title_to_recs,
               chart_labels=chart_labels, chart_values=chart_values)

@app.route("/order", methods=["GET", "POST"])
def order():
    books = scrape_books_to_scrape(2)
    if request.method == "POST":
        selected_books = request.form.getlist("selected_books")
        total = sum([book['price'] for book in books if book['title'] in selected_books])
        feedback = request.form.get("feedback")
        return f"<h3>Thank you! Your order total is ${total:.2f}. Feedback: {feedback}</h3><a href='/'>Back</a>"

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Order Your Books</title>
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                background: linear-gradient(135deg, #43cea2, #185a9d);
                color: #fff;
                padding: 20px;
            }
            nav {
                background-color: #111;
                padding: 15px;
                text-align: center;
            }
            nav a {
                color: #fff;
                margin: 0 20px;
                font-size: 18px;
                text-decoration: none;
            }
            form {
                background-color: #fff;
                color: #000;
                max-width: 600px;
                margin: auto;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            input, select, textarea {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            button {
                background-color: #28a745;
                color: #fff;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
            }
            button:hover {
                background-color: #218838;
            }
        </style>
        <script>
            function updateTotal() {
                let total = 0;
                document.querySelectorAll('input[name="selected_books"]:checked').forEach(cb => {
                    total += parseFloat(cb.getAttribute('data-price'));
                });
                document.getElementById('totalPrice').innerText = total.toFixed(2);
            }
        </script>
    </head>
    <body>
    <nav>
        <a href="/">🏠 Home</a>
        <a href="/order">🛒 Order</a>
    </nav>
    <form method="post">
        <h2>🛒 Complete Your Order</h2>
        {% for book in books %}
            <input type="checkbox" name="selected_books" value="{{ book.title }}" data-price="{{ book.price }}" onchange="updateTotal()"> {{ book.title }} - ${{ book.price }}<br>
        {% endfor %}
        <p><strong>Total: $<span id="totalPrice">0.00</span></strong></p>
        <label>Name:</label><input type="text" name="name" required>
        <label>Email:</label><input type="email" name="email" required>
        <label>Phone:</label><input type="tel" name="phone" required>
        <label>Location:</label><input type="text" name="location" required>
        <label>Delivery Option:</label>
        <select name="delivery">
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
        <label>Payment Method:</label>
        <select name="payment">
          <option value="credit_card">Credit Card</option>
          <option value="paypal">PayPal</option>
          <option value="omt">OMT</option>
          <option value="cash">Cash</option>
        </select>
        <label for="feedback">Feedback:</label>
        <textarea id="feedback" name="feedback" rows="4"></textarea>
        <button type="submit">Submit Order</button>
    </form>
    <script>
        document.querySelector('form').addEventListener('submit', function(e) {
            const selected = document.querySelectorAll('input[name="selected_books"]:checked');
            if (selected.length === 0) {
                alert("Please select at least one book before submitting.");
                e.preventDefault();
            }
        });
    </script>
    </body>
    </html>
    """, books=books)

if __name__ == "__main__":
    app.run(debug=True)
