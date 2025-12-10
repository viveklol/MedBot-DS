import streamlit as st
from app import MedicalQASystem, QueryRequest, QueryResponse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

nltk.download('punkt_tab')
USER_ICON = "assets/user.png"
BOT_ICON = "assets/chatbot.png"

# Get OpenAI API key from environment variable
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("⚠️ OPENAI_API_KEY not found in environment variables. Please create a .env file with your OpenAI API key.")
    st.stop()

qa_system = MedicalQASystem(
    openai_key=openai_key,
    vector_store_file="vector_store"
)

def main():
    # Set the title in the middle of the page
    st.set_page_config(page_title="MediBot", page_icon=":robot_face:", layout="centered")
    col1, col2, col3,col4,col5 = st.columns([1,1.4,1, 1, 1])
    with col3:
        st.image("assets/MedBot logo.png", width=60)
    st.markdown("<h1 style='text-align: center;'>MediBot - Your Medical Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Ask me any medical question!</p>", unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message['role'] == 'user':
            cols = st.columns([10, 1])
            with cols[0]:
               st.markdown(
            f"""
            <div style='width: 100%; text-align: right;'>
                <div style='border: 2px solid #800000; border-radius: 10px; padding: 10px; margin-bottom: 5px; display: inline-block;'>
                    {message['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
            with cols[1]:
                st.image(USER_ICON, width=40)

        else:
            cols = st.columns([1, 10])
            with cols[0]:
                st.image(BOT_ICON, width=40)
            with cols[1]:
                if isinstance(message['content'], dict):
                    sources_md = ""
                    if message['content']['answer'].strip() and message['content']['sources']:
                        for idx, src in enumerate(reversed(message['content']['sources']), 1):
                            if isinstance(src, dict) and 'source' in src and 'url' in src:
                                scr = src.get('score', 0) 
                                sources_md += (
                                    f'{idx}) <a href="{src["url"]}" target="_blank">{src["source"]}</a> <br>'
                                )
                            elif isinstance(src, str):
                                sources_md += f"{idx}) {src}<br>"

                    # Only calculate and display BLEU/ROUGE if sources_md is not empty
                    if sources_md:
                        bleu_score = None
                        rouge_scores = None
                        answer = message['content'].get('answer', '')
                        context = message['content'].get('context', '')
                        if answer and context:
                            reference_tokens = nltk.word_tokenize(context)
                            candidate_tokens = nltk.word_tokenize(answer)
                            bleu_score = sentence_bleu(
                                [reference_tokens],
                                candidate_tokens,
                                smoothing_function=SmoothingFunction().method1
                            )
                            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                            rouge_scores = scorer.score(context, answer)
                        st.markdown(
                            f"""
                            <div style='border: 2px solid #006666; border-radius: 10px; padding: 10px; margin-bottom: 5px; display: inline-block; text-align: left;'>
                                {message['content']['answer']}
                                <br><b>Sources:</b><br>{sources_md}
                                {"<br><b>BLEU:</b> {:.3f}".format(bleu_score) if bleu_score is not None else ""}
                                {"<br><b>ROUGE-1:</b> {:.3f} <b>ROUGE-2:</b> {:.3f} <b>ROUGE-L:</b> {:.3f}".format(
                                    rouge_scores['rouge1'].fmeasure,
                                    rouge_scores['rouge2'].fmeasure,
                                    rouge_scores['rougeL'].fmeasure
                                ) if rouge_scores is not None else ""}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        # No sources, just show the answer
                        st.markdown(
                            f"""
                            <div style='border: 2px solid #006666; border-radius: 10px; padding: 10px; margin-bottom: 5px; display: inline-block; text-align: left;'>
                                {message['content']['answer']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    prompt = st.chat_input("Enter your question:")
    if prompt:
        # Use Pydantic to validate the input
        try:
            query_request = QueryRequest(query=prompt, top_k=3)
        except Exception as e:
            st.error(f"Invalid input: {e}")
            return

        # Get the answer from the backend
        response = qa_system.get_answer(query_request.query, top_k=query_request.top_k)

        # Use Pydantic to validate the output
        try:
            query_response = QueryResponse(answer=response["answer"], sources=response["sources"],  context=response["context"] )
        except Exception as e:
            st.error(f"Invalid response from backend: {e}")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": query_response.dict()})
        st.rerun()

if __name__ == "__main__":
    main()