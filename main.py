from fastapi import FastAPI, Request,File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from llama_index import Document, GPTListIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import requests
import csv
app = FastAPI()
script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "static/")
app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")
templates = Jinja2Templates(directory="")

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "null",  # قد تحتاج إلى إضافة هذا النطاق أيضًا إذا كنت تستخدمه في التطبيق الخاص بك
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
#text classification
def zero_shot_text_classification(text,labels):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text,
        "parameters": {"candidate_labels": labels},
    })
    labels=output['labels']
    scores=output['scores']
    result=dict(zip(labels,scores))
    return result

def csv_zero_shot_classification(unlabeld_text,labels):
    labeld_data=[]
    for sentence in unlabeld_text:
        label=zero_shot_text_classification(sentence,labels)
        label = max(label, key=label.get)
        labeld_data.append({'text':sentence,'label':label})
        # افتح ملف CSV للكتابة
    # افتح ملف CSV للكتابة
    with open('labeled_data.csv', 'w', newline='') as csvfile:
        fieldnames = labeld_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # كتابة العناوين (أسماء الحقول) في الملف CSV
        writer.writeheader()

        # كتابة البيانات في الملف CSV
        writer.writerows(labeld_data)
def sentence_sentiment_analysis(sentence):
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs":sentence ,
    })
    result={}
    result[output[0][0]['label']]=output[0][0]['score']
    result[output[0][1]['label']]=output[0][1]['score']
    result[output[0][2]['label']]=output[0][2]['score']
    return result
          
def data_sentiment_analysis(unlabeld_text):
    labeld_data=[]
    pos,neg,neu=0,0,0
    for sentence in unlabeld_text:
        label=sentence_sentiment_analysis(sentence)
        label = max(label, key=label.get)
        labeld_data.append({'text':sentence,'label':label})
        if label=='positive':
            pos+=1
        if label=='negative':
            neg+=1
        if label=='neutral':
            neu+=1

        # افتح ملف CSV للكتابة
    # افتح ملف CSV للكتابة
    with open('sentiment_result.csv', 'w', newline='') as csvfile:
        fieldnames = labeld_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # كتابة العناوين (أسماء الحقول) في الملف CSV
        writer.writeheader()

        # كتابة البيانات في الملف CSV
        writer.writerows(labeld_data)
        return [pos,neg,neu]
#question answering
import os
from llama_index import Document

os.environ["OPENAI_API_KEY"] = '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'
text = """
    Next AI is the first specialized educational institute to be a center for disseminating information and exchanging knowledge in the field of artificial intelligence to qualify young students from schools and universities and prepare them to participate in this global scientific revolution.
    Next Ai center is for young students from schools and universities.
    The address of Next Ai center is: Syria, Latakia, Al-Zira'a, Opposite The University Dorms - Behind Masaya Street
    The email is: info@next-ai.org
    The phone number of the center is: 041 2491067.
    The cellphone or mobile number of the center is: 0996 238 803.
    You can register for the course that you want by filling out the following form.
    The mission of the institute will evolve and adapt according to need, with one goal only: the interest of youth, society and the country.
    Next Ai Institute’s founders are a group of competencies from inside and outside Syria.
    Next Ai center covers these concepts are: the culture of artificial intelligence, machine learning and deep learning among all age groups, to enter the market for developing applications that depend on these sciences.
    The age groups that are suitable for training in the Institute are all age groups Kids, Young, Undergraduate, Master & PHD students.
    The website of the center is: www.next-ai.org
    The courses that the center offers are: Programming with Python, Data Analysis, Machine Learning, and Deep Learning.
    The programming language that is used for AI is: Python Programming.
    The language that the center offers for AI is Python, it is language number one for machine learning, data science, and artificial intelligence.
    The Python course is for: Beginners who have never programmed before, and those who want to level up their skills, programmers switching languages to Python.
    The version of Python that the course offers is: Python 3
    The course of Python offers these concepts: define a problem, design a program to solve the problem, create executable code, and write basic unit tests.
    By the end of this Python course, the students will have gained a fundamental understanding of programming in python, dealing with AI programming tasks, applying for Python programming jobs.
    Python course focuses on the fundamental building blocks you will need to learn in order to become an AI practitioner.
    Next Ai is opened from Sunday to thursday.
    Tarek jurdi teach python course.
    Aly  teach ML course.
    """

text_list = [text]
documents = [Document(t) for t in text_list]

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

index = GPTListIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()
prompt = """
Use the Document to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer.
Question: {} 
Helpful Answer:
"""

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
@app.get("/question_answering", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("QA.html", {"request": request})
@app.get("/text_classification", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("classification.html", {"request": request})
@app.get("/sentiment_analysis", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("sentiment.html", {"request": request})


@app.get("/next-ai")
def get_next_ai(query: str):
    response = query_engine.query(prompt.format(query))
    return {"response": response}


@app.post("/set_document")
async def edit_text(new_text: str):
    global query_engine
    global prompt
    text = new_text
    text_list = [text]
    documents = [Document(t) for t in text_list]

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    index = GPTListIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    prompt = """
    Use the Document to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to
    make up an answer.
    Question: {} 
    Helpful Answer:
    """
    return {"message": "Text variable updated successfully"}
@app.get("/test_zero_text_shot_classfication")
def test_zero_text_shot(new_text:str,new_labels:str):
    new_labels=new_labels.split(',')
    result=zero_shot_text_classification(new_text,new_labels)
    print(result)
    return {"result": result}
unlabeld_text=[]
@app.post("/set_unlabeld_text")
async def set_csv(file: bytes = File(...)):
    # تحويل المحتوى النصي للملف إلى سلسلة نصية
    file_content = file.decode("utf-8")
    global unlabeld_text
    unlabeld_text=file_content.split('\n')[1:-1]
    return {"message": "تم تصنيف الـ CSV بنجاح"}
@app.get("/csv_classification")
def zero_text_shot(new_labels:str):
    new_labels=new_labels.split(',')
    csv_zero_shot_classification(unlabeld_text,new_labels)
    return {"result": 'done'}
@app.get("/save_labeled_csv")
def save(request: Request):
    # Return the CSV file as a response
    return FileResponse('labeled_data.csv', filename='labeled_data.csv')
@app.get("/save_sentiment_csv")
def save(request: Request):
    # Return the CSV file as a response
    return FileResponse('sentiment_result.csv', filename='sentiment_result.csv')
    
@app.get("/sentence_sentiment_analysis")
def sentence_sentiment_analysis_(new_text:str):
    result=sentence_sentiment_analysis(new_text)
    return {"result": result}
@app.get("/data_sentiment_analysis")
def data_sentiment_analysis_():
    result=data_sentiment_analysis(unlabeld_text)
    return {"result": result}
if __name__ == "__main__":
    uvicorn.run("main:app")
