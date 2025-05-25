import argparse, re
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM

parser = argparse.ArgumentParser(description="Инструмент для оценки качества стихотворений на основе базы знаний")
parser.add_argument("--data", type=str, required=True, help="Путь к хранилищу с тестовыми аудиозаписями")
args = parser.parse_args()
path = args.data

"""
WHISPER MODULE DEMO
"""

# Выбираем девайс: графический либо центральный процессор
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Выбираем модель Whisper из хаба HuggingFace
model_id = "openai/whisper-large-v3"

# Загружаем модель и отправляем ее на девайс
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Процессор - это "токенизатор" для аудио-моделей
processor = AutoProcessor.from_pretrained(model_id)

# Пайплайн - это сборка для распознавания речи
pipe = pipeline(
    "automatic-speech-recognition", # Тип задачи
    model=model, # Модель
    tokenizer=processor.tokenizer, # Токенизатор
    feature_extractor=processor.feature_extractor, # Извлечение признаков из спектрограмм
    torch_dtype=torch_dtype, # Тип тензоров для обработки данных
    device=device, # Девайс
    return_timestamps=True, # Временные метки
    # language='en' # Выбрать язык для перевода
)

prompt = pipe('audio.wav')['text'].strip()

"""
LLM MODULE DEMO
"""

model_id = "google/gemma-3-1b-it"
# Применяем квантизацию: загружаем модель меньшей размерности
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Инициализация модели из HuggingFace: загружается локально на наше устройство
# Это значит, что она не использует сторонние сервисы, а все вычисления выполняются у нас
model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

# Токенизация тоже производится локально, т.е. на нашем устройстве
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Определяем путь к файлу с промптом на нужном языке
prompt_path ='prompts/meta_prompt.txt'

# Открываем файл с нужным промптом
with open(prompt_path) as f:
    system_prompt = f.read()

# Подгрузка базы данных из файла
with open('data/database.json') as f:
    database = f.read()

# Системные роли удобнее подгружать из отдельного файла
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt+database},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f'USER QUERY: {prompt}'},]
        },
    ],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, temperature=0.7, top_k=50, max_new_tokens=1024)

outputs = tokenizer.batch_decode(outputs)

# Добавляем парсинг ответа модели
pattern = r'<start_of_turn>(.*?)<end_of_turn>'
matches = re.findall(pattern, outputs[0], re.DOTALL)

print(matches[1])
"""
TODO: извлечение и предобработка данных с целью оптимизации для нейросетевой обработки (нейросеть много кушает, поэтому лучше сразу сжать все файлы) 
    - описать логику для парсинга каталога с аудиозаписями
    - повытаскивать wav'ки
    - применить сжатие?
"""

# Все это надо упаковать в один *.sh, который 

# ОБРАБОТКА ВИСПЕРОМ
# - принимает на вход путь к аудиофайлу
# - запускает скрипт для whisper.cpp
# - передает выходные данные скрипта для виспера в temp 

# ФОРМИРОВАНИЕ ПРОМПТА
# скрипт, который склеивает данные виспера с системной ролью и путем к считыванию бд 
# для первых экспериментов подойдет JSON такого вида: https://github.com/vifirsanova/poetry-llm/blob/main/data/database.json
# ФОРМАТ БД: ID | META_DESCRIPTION | FUNCTION_NAME
# В промпте важно указать response_format: выбирать наиболее релевантную функцию и возвращать только ID распознанной функции в JSON 

# ОБРАБОТКА LLM
# - передаем данные из temp в скрипт для LLM 
#   - https://huggingface.co/Vikhrmodels
#   - https://huggingface.co/IlyaGusev
#   - https://huggingface.co/docs/transformers/en/quantization/gptq
# - в скрипте указываем путь 
# - удаляем данные из temp 

# 

def recognize(audio_data):
    """
    Функция, которая реализует функционал мультиагентного модуля
    :audiodata: считываем аудиофайл, преобразованным к числовым представлениям; это аудиозапись с командой юзера
    :return: ID распознанной функции 
    """

    # ОБРАБОТКА ВИСПЕРОМ
    # Обработка виспером: запускаем whisper.cpp напрямую с нужными параметрами
    #   - параметры: путь к файлу, tiny model (она хранится в каталоге), путь к выходным данным в папке temp
    
    # ФОРМИРОВАНИЕ ПРОМПТА
    # Склеивание распознанной команды с системной ролью и файлом с бд
    
    # ОБРАБОТКА LLM
    # Обработка LLM: запускаем скрипт с llama.cpp, в качестве промпта указываем путь к склеенному промпту
    # https://github.com/ggml-org/llama.cpp/tree/master/tools/run -> прописать здесь путь к входным / выходным 
    pass


# Предобработка, которая нам нужна: 
#   - квантованная модель whisper tiny
#   - квантовая llama cpp для русского языка 
#   - JSON-ка с БД-шкой
#   - промпт под мультиагент: prompt generation (DeepSeek / ChatGPT)
