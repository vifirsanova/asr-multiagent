# ASR Multiagent

Multiagent system for accessible Automated Speech Recognition (ASR) and function calling in smart home ecosystem

## Example usage

```bash
python3 script.py --data "audio.wav"
```

Sample output:

````
model
```json
{
  "ID": "SH001",
  "META_DESCRIPTION": "Controls the main lighting system in the living room",
  "FUNCTION_NAME": "living_room_light_control"
}
```
````

## Model architecture

![multiagent scheme](https://raw.githubusercontent.com/vifirsanova/asr-multiagent/refs/heads/main/scheme.svg)

---

### TODO

1. Загрузить инсталятор 
2. Запустить на сервере llama.cpp
3. Создать БД с описанием функционала 
4. Промпт-инжиниринг: просто сгенеряй промпт для выбора оптимальной функции из БД
5. Написать скрипт для формирования входного промпта под llama.cpp
