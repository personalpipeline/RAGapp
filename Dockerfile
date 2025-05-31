FROM python:3.11-slim-buster 
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0  
CMD ["python", "-c", "import gradio as gr; def greet(name): return 'Hello, ' + name + '!'; iface = gr.Interface(fn=greet, inputs='text', outputs='text'); iface.launch(server_name='0.0.0.0', server_port=7860)"]
