# AI Web Search Agent

A simple AI-powered web search agent built with Flask and Gemini LLM. This project crawls a set of seed websites, builds a local search index, and uses Google Gemini to synthesize answers from the most relevant results.

## Features

- **Web Crawler:** Crawls and stores web pages from seed URLs.
- **Search Engine:** Local search using an inverted index for fast query matching.
- **AI Synthesis:** Uses Gemini LLM to generate concise answers from top search results.
- **Modern UI:** Clean, responsive interface built with Tailwind CSS.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js (for managing dependencies, optional)
- [Google Gemini API Key](https://ai.google.dev/)

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/web-search-agent.git

   
   cd web-search-agent
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env` and add your Gemini API key:
     ```
     GEMINI_API_KEY=your_gemini_api_key
     ```

4. **Run the app:**
   ```sh
   python app.py
   ```
   The app will crawl seed URLs and start the Flask server at `http://localhost:5000`.

### Usage

- Open your browser and go to `http://localhost:5000`.
- Enter a search query to see results and an AI-generated answer.

## Project Structure

- `app.py` — Main Flask application, crawler, search, and LLM integration.
- `my_search_data.json` — Stores crawled web data.
- `templates/index.html` — Frontend UI.
- `.env` — API keys and environment variables.
- `requirements.txt` — Python dependencies.
- `vercel.json` — Vercel deployment configuration.

## Deployment

This project can be deployed to [Vercel](https://vercel.com/) using the provided `vercel.json` configuration.

## License

This project is licensed under the ISC License.

---

**Note:** This project is for educational/demo purposes. Crawling and storing third-party website data may be subject to copyright and terms
