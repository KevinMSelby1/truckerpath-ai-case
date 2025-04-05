import subprocess

print("Scraping reviews...")
subprocess.run(["python", "scripts/truckerpathscraper.py"], check=True)

print("Analyzing reviews...")
subprocess.run(["python", "scripts/reviewanalysis.py"], check=True)

print("Extracting top TF-IDF phrases...")
subprocess.run(["python", "scripts/featurecheck.py"], check=True)

print("All steps complete.")
