import logging
import uvicorn
import re
import nltk
import language_tool_python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
#from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, HfFolder
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Inicializar o corretor gramatical
tool = language_tool_python.LanguageTool('en-US')

# Carregar o modelo de parafraseamento
paraphrase_tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")

# Baixar o recurso 'punkt' da NLTK
nltk.download('punkt')

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Mensagem(BaseModel):
    mensagem: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Função para logar cada requisição e resposta de arquivos estáticos
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Requisição recebida: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Resposta enviada: {response.status_code}")
    return response

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

# Autenticação no Hugging Face
hf_api = HfApi()
token = ""  # Substitua pelo seu token real
HfFolder.save_token(token)

# Carregar o modelo e o tokenizer pré-treinados
try:
    tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-30b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-30b", trust_remote_code=True)
except Exception as e:
    logging.error(f"Erro ao carregar o modelo: {e}")
    raise

# Definir o token de preenchimento para ser o mesmo que o token de fim de sequência
tokenizer.pad_token = tokenizer.eos_token

def paraphrase(text):
    inputs = paraphrase_tokenizer.encode_plus(text, return_tensors='pt')
    outputs = paraphrase_model.generate(inputs['input_ids'], max_length=128, num_return_sequences=1, num_beams=5, early_stopping=True)
    paraphrased_text = paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text

def post_process(resposta):
    # Dividir a resposta em sentenças
    sentencas = nltk.sent_tokenize(resposta)
    
    # Remover nomes no início das sentenças e sentenças repetidas
    sentencas_unicas = []
    for sentenca in sentencas:
        # Remover nomes no início da sentença
        sentenca_sem_nome = re.sub(r'^[A-Za-z\'\-\s]*:', '', sentenca)
        
        # Verificar se a sentença (sem o nome) é repetida ou similar
        if sentenca_sem_nome not in sentencas_unicas:
            sentencas_unicas.append(sentenca_sem_nome)
    
    # Juntar as sentenças únicas de volta em uma única string
    resposta = ' '.join(sentencas_unicas)
    
    # Remover possíveis duplicatas consecutivas de palavras ou frases
    resposta = re.sub(r'\b(\w+)( \1\b)+', r'\1', resposta)
    
    # Correção gramatical e ortográfica
    resposta_corrigida = tool.correct(resposta)
    
    # Parafrasear para melhorar a fluidez
    resposta_final = paraphrase(resposta_corrigida)
    
    return resposta_final.strip()

@app.post('/responder')
async def responder(dados: Mensagem):
    try:
        logging.info(f"Recebido dados: {dados}")
        
        # Adicionar um prefixo à mensagem do usuário para indicar que é uma pergunta
        mensagem_com_prefixo = "Pergunta: " + dados.mensagem
        
        # Codificar a mensagem do usuário com prefixo e gerar uma resposta
        inputs = tokenizer.encode_plus(
            mensagem_com_prefixo,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=150, 
            return_attention_mask=True
        )
        
        logging.info(f"Inputs gerados: {inputs}")
        
        # Gerar uma resposta
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=40, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,  # Ajustar para gerar respostas mais variadas
            top_p=0.85  # Ajustar para reduzir a repetição
        )
        
        logging.info(f"Outputs gerados: {outputs}")
        
        # Decodificar a resposta, removendo o texto da entrada
        resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        resposta = resposta[len(mensagem_com_prefixo):].strip()
        
        # Pós-processamento da resposta
        resposta = post_process(resposta)
        
        logging.info(f"Resposta gerada e pós-processada: {resposta}")
        
        return JSONResponse(content={'resposta': resposta})
    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host='0.0.0.0', port=8000)
