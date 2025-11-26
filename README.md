# Movie Recommendations

A movie recommendation system that uses semantic search to find similar movies based on a query title.

## Setup

### 1. Get API Keys

You'll need two API keys:

**TMDB API Key:**
1. Go to [The Movie Database](https://www.themoviedb.org/)
2. Create an account and navigate to your API settings
3. Request an API key
4. Copy your API key

**OpenAI API Key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API keys section
4. Create a new API key
5. Copy your API key

### 2. Configure Environment Variables

Create a `.env` file in the project root with your API keys:

```
TMDB_API_KEY=your_tmdb_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install metaflow openai faiss-cpu python-dotenv requests numpy
```

## Usage

Run the movie recommendation flow with:

```bash
python3 flow.py run --num_pages 5 --query_title "Arrival" --top_k 5
```

**Parameters:**
- `--num_pages`: Number of pages of movies to fetch from TMDB (default: 5)
- `--query_title`: The movie title to find recommendations for (default: "Arrival")
- `--top_k`: Number of recommendations to return (default: 5)

**Note:** Use underscores (not dashes) for parameter names in the CLI command.

## Example Output

When you run the flow, you'll see recommendations and an explanation like this:

```
In terms of mood and tone, these movies balance tension and introspection, allowing for moments of suspense while also delving into the characters' emotional journeys. *Predator* and *War of the Worlds* bring a thrilling edge with their action-packed sequences, but they also explore the fear of the unknown and humanity's response to it, similar to *Arrival*. The pacing of these films often builds steadily, drawing viewers into their worlds before launching into intense climaxes, which keeps you engaged and on the edge of your seat. Plus, they all fall within the realm of science fiction, blending adventure and existential themes, making them a great fit for anyone who enjoys thought-provoking narratives with a touch of excitement.
```
