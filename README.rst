### Docker build 

docker build -t pdfqa .

### Docker Run

docker run -e OPENAI_API_KEY='' -e GROQ_KEY='' -p 8000:8000 pdfqa
