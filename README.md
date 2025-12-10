# MedBot - Medical Question Answering System

MedBot is an AI-powered medical question answering system that uses retrieval-augmented generation (RAG) to provide accurate medical information. The system leverages OpenAI's GPT models combined with a vector database of medical knowledge from the MedQuAD dataset.

## Features

- **Intelligent Q&A**: Ask medical questions and get accurate, source-backed answers
- **Vector Search**: Uses FAISS for efficient semantic search over medical documents
- **Source Attribution**: Every answer includes references to the source medical documents
- **Interactive UI**: Beautiful Streamlit-based web interface
- **Performance Metrics**: Displays BLEU and ROUGE scores for answer quality evaluation

## Project Structure

```
MedicalQA-main/
├── app.py                          # FastAPI backend with QA system logic
├── streamui.py                     # Streamlit web interface
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── medquad_data.csv               # MedQuAD medical dataset (~26MB)
├── vector_store/                  # Pre-built FAISS vector database (~90MB)
│   ├── index.faiss
│   └── index.pkl
└── assets/                        # UI assets (images, logos)
```

## Prerequisites

- Python 3.9 or higher (tested with Python 3.9.20 and 3.12.3)
- OpenAI API key
- 4GB+ RAM (for loading the vector store and models)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RITHVIK23/Medquad-chatbot.git
cd Medquad-chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
```

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=your-openai-api-key-here
```

You can get an OpenAI API key from: https://platform.openai.com/api-keys

## Running the Application

### Option 1: Streamlit Web Interface (Recommended)

Run the Streamlit app for an interactive web interface:

```bash
streamlit run streamui.py
```

The app will open in your default browser at `http://localhost:8501`

### Option 2: Command Line Interface

Run the app.py directly for CLI interaction:

```bash
python app.py
```

Then type your medical questions directly in the terminal.

### Option 3: FastAPI Server

To run as a REST API server:

```bash
uvicorn app:app --reload --port 8000
```

Access the API at `http://localhost:8000/docs` for the interactive API documentation.

## Usage

1. **Start the application** using one of the methods above
2. **Ask a medical question** (e.g., "What are the symptoms of diabetes?")
3. **View the answer** along with source references and quality metrics
4. **Check sources** by clicking on the provided links (if using web interface)

## Technical Details

### Components

- **Embeddings**: Uses `all-MiniLM-L6-v2` model from HuggingFace for document embeddings
- **Vector Store**: FAISS with cosine similarity for efficient retrieval
- **LLM**: OpenAI GPT models for answer generation
- **Framework**: FastAPI for backend, Streamlit for frontend
- **Evaluation**: BLEU and ROUGE metrics for answer quality

### Data

The system uses the **MedQuAD** (Medical Question Answering Dataset) which contains:
- Medical questions and answers from trusted sources
- Pre-processed and embedded in FAISS vector store
- Covers various medical topics and conditions

## Troubleshooting

### "OPENAI_API_KEY not found" Error

Make sure you have:
1. Created a `.env` file in the project root
2. Added your OpenAI API key: `OPENAI_API_KEY=sk-...`
3. The key is valid and has available credits

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory Issues

If you encounter memory issues:
- Close other applications
- The vector store requires ~2GB RAM to load
- Consider using a machine with more RAM

### Mac M1/M2 Users

If you're using Mac M1/M2, run this code using Rosetta. Instructions: https://support.apple.com/en-us/102527

## License

This project uses the MedQuAD dataset. Please refer to the original dataset license for usage terms.

## Acknowledgments

- MedQuAD Dataset: https://github.com/abachaa/MedQuAD
- LangChain for RAG framework
- OpenAI for language models
- HuggingFace for embedding models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
