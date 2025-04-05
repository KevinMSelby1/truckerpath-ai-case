# Import necessary libraries
from google_play_scraper import app, reviews
import pandas as pd
import csv
import time
import random

# Function to scrape reviews with rate-limiting consideration and stop after 5000 reviews
def scrape_reviews_limited(package_name, max_reviews=3620, min_delay=2, max_delay=5):
    reviews_data = []  # List to store all reviews
    continuation_token = None  # Initialize continuation_token for pagination
    total_reviews = 0  # Counter for total reviews scraped

    while total_reviews < max_reviews:
        # Get reviews using google-play-scraper's reviews method
        result, continuation_token = reviews(
            package_name,
            count=200,  # You can adjust the batch size here
            lang='en',
            country='us',
            filter_score_with=1,  # Only 1-star reviews
            continuation_token=continuation_token  # Pagination support
        )

        reviews_data.extend(result)
        total_reviews += len(result)
        print(f"Collected {total_reviews} reviews so far...")

        if total_reviews >= max_reviews:
            reviews_data = reviews_data[:max_reviews]
            break

        if not continuation_token:
            break  # No more reviews to fetch

        delay = random.uniform(min_delay, max_delay)
        print(f"Sleeping for {delay:.2f} seconds to respect rate limits...")
        time.sleep(delay)

    return reviews_data

# Function to save the reviews data to a CSV file
def save_reviews_to_csv(reviews_data, filename='reviews.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = reviews_data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews_data)
    print(f"Reviews saved to {filename}")

# Example usage
package_name = 'com.sixdays.truckerpath'
all_reviews_data = scrape_reviews_limited(package_name, max_reviews=3620)

# Save the reviews to a CSV file in /app/data/
save_reviews_to_csv(all_reviews_data, filename='/app/data/truckerpath_1star_3620.csv')

# Print the first few reviews to check
for review in all_reviews_data[:5]:
    print(f"User: {review['userName']}")
    print(f"Rating: {review['score']}")
    print(f"Review: {review.get('content', 'N/A')}")
    print(f"Date: {review.get('at', 'N/A')}")
    print("-" * 40)

