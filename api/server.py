import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["ARGOS_PACKAGES_DIR"] = os.path.join(BASE_DIR, "argos", "models")

import collections
import json
from time import time, localtime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import logging
import webrtcvad
import argostranslate.translate
import edge_tts

from faster_whisper import WhisperModel
import numpy as np
import asyncio
import torch

import io
import glob
import wave
from piper import PiperVoice, SynthesisConfig

from transformers import MarianMTModel, MarianTokenizer
from collections import OrderedDict

import gc

# logging.basicConfig(filename=None, )
logger = logging.getLogger(__name__)
LOG_LEVEL = logging.DEBUG

logger.setLevel(LOG_LEVEL)

log_format = "%(levelname)s - %(message)s"

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(logging.Formatter(log_format))

logger.addHandler(console_handler)

SAMPLE_RATE = 16000
VAD_FRAME_MS = 30        # ms
PREBUFFER_DURATION = 1 # Sekunden
POST_SPEECH_SILENCE = 0.5 # Sekunden
PARTIAL_AUDIO_THRESHOLD = 2.75 # Sekunden
CHUNK_SIZE = 3200
VOICES = {
    # "ar": {"male": "ar-AE-HamdanNeural", "female": "ar-AE-FatimaNeural"},
    "zh": {"male": "zh-CN-YunjianNeural", "female": "zh-CN-XiaoxiaoNeural"},
    "de": {"male": "de-AT-JonasNeural", "female": "de-AT-IngridNeural"},
    "en": {"male": "en-AU-WilliamMultilingualNeural", "female": "en-AU-NatashaNeural"},
    "es": {"male": "es-AR-TomasNeural", "female": "es-AR-ElenaNeural"},
    "fr": {"male": "fr-BE-GerardNeural", "female": "fr-BE-CharlineNeural"},
    "it": {"male": "it-IT-DiegoNeural", "female": "it-IT-ElsaNeural"},
    "ja": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
    "ko": {"male": "ko-KR-InJoonNeural", "female": "ko-KR-SunHiNeural"},
    "nl": {"male": "nl-NL-MaartenNeural", "female": "nl-NL-ColetteNeural"},
    "pl": {"male": "pl-PL-MarekNeural", "female": "pl-PL-ZofiaNeural"},
    "pt": {"male": "pt-BR-AntonioNeural", "female": "pt-BR-FranciscaNeural"},
    "ru": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"},
    "sv": {"male": "sv-SE-MattiasNeural", "female": "sv-SE-SofieNeural"},
    "tr": {"male": "tr-TR-AhmetNeural", "female": "tr-TR-EmelNeural"},
    "uk": {"male": "uk-UA-OstapNeural", "female": "uk-UA-PolinaNeural"},
    "vi": {"male": "vi-VN-NamMinhNeural", "female": "vi-VN-HoaiMyNeural"}
}
LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "zh": "Chinese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "es": "Spanish",
    "pt": "Portuguese",
    "tr": "Turkish",
    "it": "Italian",
    "ja": "Japanese",
    "sv": "Swedish",
    "pl": "Polish",
    "ko": "Korean",
    "nl": "Dutch",
    "vi": "Vietnamese",
}

PIPER_MODELS_DIR = os.path.join(BASE_DIR, "piper", "models")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = DEVICE == "cuda"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

torch.set_grad_enabled(False)

# Manche Sätze sind im Whisper Modell übertrainiert. Damit diese aktzeptiert werden wird eine höhere confidence benötigt.
OVERTRAINED_PHRASES = ["Vielen Dank.", "Thank you.", "Bye.", "Tschüss.", "Merci.", "Gracias.", "Grazie.", "Спасибо.", "Amen."]

def _find_piper_model(lang: str) -> str | None:
    pattern = os.path.join(PIPER_MODELS_DIR, lang, "*.onnx")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

_available = [l for l in os.listdir(PIPER_MODELS_DIR) 
              if os.path.isdir(os.path.join(PIPER_MODELS_DIR, l))] \
             if os.path.isdir(PIPER_MODELS_DIR) else []
logger.info(f"Piper-Modelle verfügbar: {_available or 'keine'}")

whisper_model = WhisperModel("turbo", device="cuda", compute_type="float16") \
    if torch.cuda.is_available() \
    else WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8)

class LiveTranslationSession:
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        self.model = whisper_model
        
        self.vad = webrtcvad.Vad(3)
        
        self.STATE = "WAITING"
        self.last_speech_time = None
        self.speech_start_time = None
        self.audio_buffer = []
        self.partial_result_sent = False
        # self.empty_result_since = None
        
        max_prebuffer_frames = int(PREBUFFER_DURATION * SAMPLE_RATE * 2 / CHUNK_SIZE)
        self.prebuffer = collections.deque(maxlen=max_prebuffer_frames)

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    def process_audio_chunk(self, audio_bytes: bytes):
        results = []
        
        # logger.debug(f"State: {self.STATE}")
        
        if self.STATE == "WAITING":
            self.prebuffer.append(audio_bytes)
            frame_len = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2
            for i in range(0, len(audio_bytes), frame_len):
                frame = audio_bytes[i:i + frame_len]
                if len(frame) == frame_len and self.vad.is_speech(frame, SAMPLE_RATE):
                    self.STATE = "TRANSCRIBING"
                    self.audio_buffer = list(self.prebuffer)
                    break

        elif self.STATE == "TRANSCRIBING":
            self.audio_buffer.append(audio_bytes)
            self.speech_start_time = time() if self.speech_start_time is None else self.speech_start_time

            # VAD
            frame_len = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2
            for i in range(0, len(audio_bytes), frame_len):
                frame = audio_bytes[i:i + frame_len]
                if len(frame) == frame_len and self.vad.is_speech(frame, SAMPLE_RATE):
                    self.last_speech_time = time()

            if self.last_speech_time and (time() - self.last_speech_time) > POST_SPEECH_SILENCE:
                
                combined = b"".join(self.audio_buffer)
                audio_float = self._bytes_to_float32(combined)
                audio_duration = len(audio_float) / SAMPLE_RATE
                if audio_duration < 1.2:
                    # zu kurz, wahrscheinlich Fehltrigger -> Reset ohne Transkription
                    self.STATE = "WAITING"
                    self.audio_buffer = []
                    self.prebuffer.clear()
                    self.partial_result_sent = False
                    self.speech_start_time = None
                    self.last_speech_time = None
                    return []

                segments, _ = self.model.transcribe(
                    audio_float,
                    language=self.source_lang,
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    # no_speech_threshold=0.25,
                    # log_prob_threshold=-0.7
                )
                segments = list(segments)
                texts = []
                for segment in segments:
                    segment_text = segment.text.strip()
                    if segment.avg_logprob < -0.95:
                        continue
                    if segment_text in OVERTRAINED_PHRASES and segment.avg_logprob < -0.425:
                        logger.debug(f"Partial segment '{segment_text}' filtered out (avg_logprob: {segment.avg_logprob:.2f})")
                        continue
                    texts.append(segment_text)
                text = " ".join(t for t in texts).strip()
                    
                print_debug_info(segments)

                if text:
                    results.append({"type": "final", "text": text, "audio_duration": audio_duration})
                    logger.debug(f"Audio duration: {audio_duration:.2f}s")
                    

                # Reset
                self.STATE = "WAITING"
                self.audio_buffer = []
                self.prebuffer.clear()
                self.partial_result_sent = False
                self.speech_start_time = None
                self.last_speech_time = None
                
            if self.speech_start_time and (time() - self.speech_start_time) > PARTIAL_AUDIO_THRESHOLD:
                combined = b"".join(self.audio_buffer)
                audio_float = self._bytes_to_float32(combined)
                audio_duration = len(audio_float) / SAMPLE_RATE
                if audio_duration < 1.2:
                    # zu kurz, wahrscheinlich Fehltrigger -> Reset ohne Transkription
                    self.STATE = "WAITING"
                    self.audio_buffer = []
                    self.prebuffer.clear()
                    self.partial_result_sent = False
                    self.speech_start_time = None
                    self.last_speech_time = None
                    return []

                segments, _ = self.model.transcribe(
                    audio_float,
                    language=self.source_lang,
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    # no_speech_threshold=0.2,
                    # log_prob_threshold=-0.625
                )
                segments = list(segments)
                texts = []
                for segment in segments:
                    segment_text = segment.text.strip()
                    if segment.avg_logprob < -0.95:
                        continue
                    if segment_text in OVERTRAINED_PHRASES and segment.avg_logprob < -0.425:
                        logger.debug(f"Partial segment '{segment_text}' filtered out (avg_logprob: {segment.avg_logprob:.2f})")
                        continue
                    texts.append(segment_text)
                text = " ".join(t for t in texts).strip()
                    
                print_debug_info(segments)

                
                if text:
                    logger.debug(f"Audio duration: {audio_duration:.2f}s")
                    if not self.partial_result_sent:
                        results.append({"type": "partial", "text": text, "audio_duration": audio_duration})
                        self.partial_result_sent = True
                    else:
                        results.append({"type": "final", "text": text, "audio_duration": audio_duration})
                        self.partial_result_sent = False
                        self.STATE = "WAITING"
                        self.audio_buffer = []
                        self.prebuffer.clear()
                        self.last_speech_time = None
                        
                    # self.speech_start_time = time()
                    self.speech_start_time = time() - (PARTIAL_AUDIO_THRESHOLD - 1.5)
                else:
                    self.reset()
                    

        return results
    
    def reset(self):
        self.STATE = "WAITING"
        self.audio_buffer = []
        self.prebuffer.clear()
        self.last_speech_time = None
        self.speech_start_time = None
        self.partial_result_sent = False


def print_debug_info(segments):
    logger.debug("*"*30)
    for segment in segments:
        logger.debug(f"segment avg_logprob: {segment.avg_logprob}")
    logger.debug("*"*30)


class TranslationRequest:
    """Represents a single translation request with metadata."""
    def __init__(self, text: str, source_lang: str, target_lang: str, use_opus: bool, request_id: str = None):
        self.text = text
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_opus = use_opus
        self.request_id = request_id or str(time())
        self.result = None
        self.error = None
        self.done_event = asyncio.Event()


class TranslationWorker:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.worker_task = None
        
    async def start(self):
        if not self.is_running:
            self.is_running = True
            self.worker_task = asyncio.create_task(self._process_queue())
            logger.debug("Global translation worker started")
    
    async def stop(self):
        self.is_running = False
        if self.worker_task:
            try:
                await asyncio.wait_for(self.worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.worker_task.cancel()
            logger.debug("Global translation worker stopped")
    
    async def queue_translation(self, request: TranslationRequest):
        await self.queue.put(request)
        await request.done_event.wait()
        
        if request.error:
            raise Exception(f"Translation error: {request.error}")
        return request.result
    
    async def _process_queue(self):
        try:
            while self.is_running:
                try:
                    request = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    
                    try:
                        logger.debug(f"Processing translation: {request.source_lang}->{request.target_lang}")
                        
                        result = translate(
                            request.text,
                            request.source_lang,
                            request.target_lang,
                            request.use_opus
                        )
                        request.result = result
                        
                        # logger.debug(f"Translation completed (ID: {request.request_id})")
                        logger.debug(f"Translation completed")
                        
                    except Exception as e:
                        request.error = str(e)
                        logger.error(f"Translation failed (ID: {request.request_id}): {str(e)}")
                    finally:
                        request.done_event.set()
                        self.queue.task_done()
                        
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            logger.error(f"Translation worker error: {str(e)}")
        finally:
            self.is_running = False


app = FastAPI()
translation_worker = TranslationWorker()


@app.on_event("startup")
async def startup_event():
    await translation_worker.start()


@app.on_event("shutdown")
async def shutdown_event():
    await translation_worker.stop()


@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    
    test_data = [] 
    
    await websocket.accept()
    session = None
    
    try:
        init_data = await websocket.receive_json()
        source_lang = init_data.get("source_lang", "de")
        target_lang = init_data.get("target_lang", "en")
        voice_gender = init_data.get("voice_gender", "male")
        use_opus = init_data.get("use_better_translation", False)
        
        logger.debug(f"languages: {source_lang} to {target_lang}")
        logger.debug(f"translation model: {'Opus-MT' if use_opus else 'Argos'}")
                
        await websocket.send_json({
            "type": "connected",
            "message": "server connection established"
        })
        
        logger.debug("warming up translator")
        max_retries = 6
        for attempt in range(max_retries):
            try:
                translate("warmup", source_lang, target_lang, use_opus)
                logger.debug("translator warmup complete")
                break
            except Exception as e:
                logger.warning(f"Warmup attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(.25)
                
        logger.debug("Warming up Piper")
        voice = get_piper_voice(target_lang)
        voice.synthesize("warmup")
        logger.debug("Piper warmup complete")
        
        session = LiveTranslationSession(source_lang, target_lang)
        
        await websocket.send_json({
            "type": "ready",
            "message": "Session initialized"
        })
        
        while True:
            test_data_map = {
                "audio_duration": 0,
                "STT": 0,
                "MT": 0,
                "TTS": 0,
                "total": 0
            }
            data = await websocket.receive()
            
            if data["type"] == "websocket.receive":
                if "bytes" in data and data["bytes"]:
                    audio_bytes = data["bytes"]
                    
                    start_time = time()
                    results = session.process_audio_chunk(audio_bytes)
                    stt_time = time() - start_time
                    if stt_time >= 0.01:
                        logger.debug(f"STT Time: {stt_time}")
                        test_data_map["STT"] = stt_time
                    
                    for result in results:
                        await websocket.send_json(result)
                        
                        if result["type"] == "final":
                            test_data_map["audio_duration"] = result["audio_duration"]
                            final = result["text"]
                            logger.info(f"Erkannter Text: {final}")
                            translate_start_time = time()
                            
                            # Queue translation request through the worker
                            trans_request = TranslationRequest(
                                text=final.capitalize(),
                                source_lang=source_lang,
                                target_lang=target_lang,
                                use_opus=use_opus
                            )
                            translated = await translation_worker.queue_translation(trans_request)
                            
                            mt_time = time() - translate_start_time
                            logger.debug(f"MT Time: {mt_time}")
                            test_data_map["MT"] = mt_time
                            
                            tts_start_time = time()
                            tts_audio = await generate_tts(translated, target_lang, voice_gender)  
                            tts_time = time() - tts_start_time
                            logger.debug(f"TTS Time: {tts_time}")
                            test_data_map["TTS"] = tts_time
                            
                            logger.debug("Sending audio tts result...")
                            await websocket.send_bytes(tts_audio)
                            logger.debug("Audio tts result sent")
                            
                            logger.debug("Sending translated text result...")
                            logger.info(f"Übersetzter Text: {translated}")

                            await websocket.send_json({
                                "type": "translated",
                                "text": translated
                            })
                            time_took = time() - start_time
                            logger.debug(f"time took: {time_took}")
                            test_data_map["total"] = time_took
                            logger.debug("Text result sent")
                            test_data.append(test_data_map)
                            logger.info("-"*20)
                        elif result["type"] == "partial":
                            partial = result["text"]
                            logger.info(f"Erkannter Partial Text: {partial}")
                            # trans_request = TranslationRequest(
                            #     text=final.capitalize(),
                            #     source_lang=source_lang,
                            #     target_lang=target_lang,
                            #     use_opus=use_opus
                            # )
                            # translated = await translation_worker.queue_translation(trans_request)
                            
                            # tts_audio = await generate_tts(translated, target_lang, voice_gender)  
                            
                            # logger.debug("Sending audio tts result...")
                            # await websocket.send_bytes(tts_audio)
                            # logger.debug("Audio tts result sent")
                            
                            # logger.debug("Sending translated text result...")
                            # logger.debug(f"Übersetzter Partial Text: {final}")
                            # await websocket.send_json({
                            #     "type": "translated",
                            #     "text": translated
                            # })
                            
                            # logger.debug("Text result sent")      
                            
                elif "text" in data and data["text"]:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "reset":
                        session.reset()
                        logger.debug("Session reset by client")
                    
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        logger.error(f"Error: {str(e.with_traceback())}")
    
    # print(f"Test Data: {test_data}")
    # open("test_data.json", "w").write(json.dumps(test_data))


# async def generate_tts(text: str, lang: str, voice_gender: str) -> bytes:
#     voice = VOICES.get(lang, "en").get(voice_gender, "male")
#     communicate = edge_tts.Communicate(text, voice, rate="+10%")
#     audio_chunks = []
#     async for chunk in communicate.stream():
#         if chunk["type"] == "audio":
#             audio_chunks.append(chunk["data"])

#     return b"".join(audio_chunks)


# Model-Cache
_piper_cache: dict[str, PiperVoice] = {}

def get_piper_voice(lang: str) -> PiperVoice | None:
    if lang in _piper_cache:
        return _piper_cache[lang]
    model_path = _find_piper_model(lang)
    if model_path is None:
        return None
    voice = PiperVoice.load(model_path)
    _piper_cache[lang] = voice
    return voice


async def generate_tts(text: str, lang: str, voice_gender: str) -> bytes:
    voice = get_piper_voice(lang)

    if voice is not None:
        # blocking -> Executor
        loop = asyncio.get_event_loop()
        def _synth():
            buf = io.BytesIO()
            for chunk in voice.synthesize(text):
                buf.write(chunk.audio_int16_bytes)
            return buf.getvalue()
        pcm = await loop.run_in_executor(None, _synth)

        # PCM -> WAV #https://www.programmersought.com/article/935611585312/
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)       # int16
            wf.setframerate(voice.config.sample_rate)
            wf.writeframes(pcm)
        logger.debug(f"TTS: Piper ({lang})")
        return wav_buf.getvalue()

    else:
        # Fallback edge-tts
        logger.debug(f"TTS: edge-tts Fallback ({lang})")
        edge_voice = VOICES.get(lang, VOICES["en"]).get(voice_gender, "male")
        communicate = edge_tts.Communicate(text, edge_voice, rate="+10%")
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

@app.post("/transcript")
async def generate_transcript(
    lang: str = Form(...),
    audio: UploadFile = File(...)
):
    logger.debug(f"Lang: {lang}")

    audio_bytes = await audio.read()
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    duration_seconds = len(audio_np) / 16000
    estimated_time = round(duration_seconds * 0.1, 0) if torch.cuda.is_available() else round(duration_seconds * 0.95, 0)

    if lang == "auto":
        lang = None

    async def event_stream():
        yield f"data: {json.dumps({'type': 'estimated_time', 'seconds': estimated_time})}\n\n"
        await asyncio.sleep(0)  # Control an Event Loop abgeben -> nicht stuck?

        model = whisper_model if torch.cuda.is_available() else WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=8)
        
        # transcribe blocking -> executro
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None, 
            lambda: model.transcribe(audio_np, beam_size=1, language=lang, vad_filter=True, best_of=1, temperature=[0.0, 0.2], task="transcribe")
        )

        transcript = ""
        for segment in segments:
            transcript += segment.text.lstrip() + "\n"

        yield f"data: {json.dumps({'type': 'transcript', 'text': transcript.strip()})}\n\n"
        await asyncio.sleep(0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def translate(text: str, src: str, tgt: str, use_opus: bool):
    if text.isnumeric(): #opus hat Probleme mit reinen Zahlen, überspringe MT in diesem Fall
        return text
    if use_opus:
        opus_model = get_opus_model_name(src, tgt)
        if opus_model != "":
            return  translate_opus(opus_model, text, tgt)
        else:
            opus_model = get_opus_model_name(src, "en")
            english_translation = translate_opus(opus_model, text, "en")
            opus_model = get_opus_model_name("en", tgt)
            return translate_opus(opus_model, english_translation, tgt)
    else:
        return translate_argos(text, src, tgt)


def translate_argos(text: str, source_lang: str, target_lang: str, append_punctuation: bool = False) -> str:
    text += "." if append_punctuation else ""
    result = argostranslate.translate.translate(text, source_lang, target_lang)
    return result if not append_punctuation else result[:-1]


@app.post("/translate")
async def get_translation(
    text: str = Form(...), 
    source_lang: str = Form(...),
    target_lang: str = Form(...),
):
    if len(text.strip()) < 2:
        return {"translated": text}
    return {"translated": translate(text, source_lang, target_lang, use_opus=True)}
    # opus_model = get_opus_model_name(source_lang, target_lang)
    # logger.debug(f"Opus Model: {opus_model}")
    # if opus_model != "":
    #     logger.debug("Translating with opus")
    #     return {"translated": translate_opus(opus_model, text, target_lang)}
    # else:
    #     logger.debug("Translating with argos")
    #     append_punctuation = True if text[-1] not in [".", "?", ",", "!"] else False
    #     return {"translated": translate_argos(text, source_lang, target_lang, append_punctuation)}


MAX_MODELS = 5
model_cache = OrderedDict()

def get_model(model_name):
    if model_name in model_cache:
        model_cache.move_to_end(model_name)
        return model_cache[model_name]
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE).eval()

    if len(model_cache) >= MAX_MODELS:
        old_name, (old_tok, old_model) = model_cache.popitem(last=False)
        del old_tok
        del old_model
        gc.collect()
        if USE_CUDA:
            torch.cuda.empty_cache()

    model_cache[model_name] = (tokenizer, model)
    return tokenizer, model

def translate_opus(model_name, q, tgt):
    if needs_prefix(model_name):
        q = get_token(tgt) + q
    tokenizer, model = get_model(model_name)
    
    tokens = tokenizer(
        q,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.inference_mode():
        translated = model.generate(
            **tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            max_length=None
        )
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def get_opus_model_name(src, tgt):
    langs_to_model = {
        ("en","de"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("en","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-fr",
        ("en","zh"): "opus_mt_models/Helsinki-NLP__opus-mt-en-zh",
        ("en","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-zle", #mt-en-ru
        ("en","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-zle", #mt-en-uk
        ("en","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-es",
        ("en","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-pt",
        ("en","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-tr",
        ("en","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-it",
        ("en","ja"): "opus_mt_models/Helsinki-NLP__opus-tatoeba-en-ja", # opus_mt_models/opus-tatoeba-en-ja 
        ("en","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-gmq", #opus-mt-en-sv
        ("en","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-en-zlw", # / opus-mt-tc-big-zlw-en / opus-mt-en-pl
        ("en","ko"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-en-ko",
        ("en","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw", #opus-mt-en-nl
        ("en","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-en-vi", 
        
        ("de","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("de","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-de-fr",
        ("de","zh"): "",
        ("de","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-de-zle",
        ("de","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-de-zle",
        ("de","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-de-es",
        ("de","pt"): "",
        ("de","tr"): "",
        ("de","it"): "opus_mt_models/Helsinki-NLP__opus-mt-de-it",
        ("de","ja"): "",  #back track?
        ("de","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-de-gmq",
        ("de","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-de-pl",
        ("de","ko"): "",
        ("de","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("de","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-de-vi",
        
        ("fr","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-fr-en",
        ("fr","de"): "opus_mt_models/Helsinki-NLP__opus-mt-fr-de",
        ("fr","zh"): "",
        ("fr","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-fr-zle", #test
        ("fr","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-fr-zle",
        ("fr","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("fr","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("fr","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-tr",
        ("fr","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("fr","ja"): "",  #back track?
        ("fr","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-fr-sv", ################################################################
        ("fr","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-fr-pl",
        ("fr","ko"): "",
        ("fr","nl"): "",
        ("fr","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-fr-vi",
        
        ("ru","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-en",
        ("ru","de"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-de",
        ("ru","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-fr",
        ("ru","zh"): "",
        ("ru","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-zle",
        ("ru","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-es",
        ("ru","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-pt",
        ("ru","tr"): "",
        ("ru","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-it",
        ("ru","ja"): "",  #back track?
        ("ru","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-gmq",
        ("ru","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-zlw",
        ("ru","ko"): "",
        ("ru","nl"): "",
        ("ru","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-ru-vi", 
        
        ("uk","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-en",
        ("uk","de"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-de",
        ("uk","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-fr",
        ("uk","zh"): "",
        ("uk","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-zle",
        ("uk","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-es",
        ("uk","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-pt",
        ("uk","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-base-uk-tr",
        ("uk","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-it",
        ("uk","ja"): "",  #back track?
        ("uk","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-gmq",
        ("uk","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zle-zlw",
        ("uk","ko"): "",
        ("uk","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-uk-nl",
        ("uk","vi"): "", 
        
        ("es","en"): "opus_mt_models/Helsinki-NLP__opus-mt-es-en",
        ("es","de"): "opus_mt_models/Helsinki-NLP__opus-mt-es-de",
        ("es","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("es","zh"): "",
        ("es","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-es-zle",
        ("es","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-es-zle",
        ("es","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("es","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-tr",
        ("es","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("es","ja"): "", #back track?
        ("es","sv"): "",
        ("es","pl"): "opus_mt_models/Helsinki-NLP__opus-mt-es-pl",
        ("es","ko"): "",
        ("es","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-es-nl",
        ("es","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-es-vi", 
        
        ("nl","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("nl","de"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmw-gmw",
        ("nl","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-nl-fr",
        ("nl","es"): "opus_mt_models/Helsinki-NLP__opus-mt-nl-es",
        ("nl","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-nl-uk",
        ("nl","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-nl-sv",
        ("nl","it"): "",
        ("nl","pt"): "",
        ("nl","ru"): "",
        ("nl","pl"): "",
        ("nl","tr"): "",
        ("nl","zh"): "",
        ("nl","ja"): "", #back track wie?
        ("nl","ko"): "",
        ("nl","vi"): "",

        ("pt","en"): "opus_mt_models/Helsinki-NLP__opus-mt-itc-en",
        ("pt","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("pt","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("pt","it"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("pt","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-pt-zle",
        ("pt","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-pt-zle",
        ("pt","de"): "",
        ("pt","sv"): "",
        ("pt","pl"): "",
        ("pt","nl"): "",
        ("pt","tr"): "",
        ("pt","zh"): "",
        ("pt","ja"): "", #back track wie?
        ("pt","ko"): "",
        ("pt","vi"): "",

        ("tr","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-tr-en",
        ("tr","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tr-fr",
        ("tr","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tr-es",
        ("tr","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-base-tr-uk",
        ("tr","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-tr-sv",
        ("tr","de"): "",
        ("tr","it"): "",
        ("tr","pt"): "",
        ("tr","ru"): "",
        ("tr","pl"): "",
        ("tr","nl"): "",
        ("tr","zh"): "",
        ("tr","ja"): "", #back track wie?
        ("tr","ko"): "",
        ("tr","vi"): "",

        ("it","en"): "opus_mt_models/Helsinki-NLP__opus-mt-itc-en",
        ("it","de"): "opus_mt_models/Helsinki-NLP__opus-mt-it-de",
        ("it","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("it","es"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("it","pt"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-itc",
        ("it","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-it-zle",
        ("it","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-it-zle",
        ("it","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-it-sv",
        ("it","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-itc-tr",
        ("it","vi"): "opus_mt_models/Helsinki-NLP__opus-mt-it-vi",
        ("it","ja"): "", #back track wie?
        ("it","pl"): "",
        ("it","nl"): "",
        ("it","zh"): "",
        ("it","ko"): "",

        ("sv","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmq-en",
        ("sv","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-fr",
        ("sv","es"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-es",
        ("sv","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-ru",
        ("sv","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-uk",
        ("sv","nl"): "opus_mt_models/Helsinki-NLP__opus-mt-sv-nl",
        ("sv","tr"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-gmq-tr",
        ("sv","zh"): "",
        ("sv","de"): "",
        ("sv","it"): "",
        ("sv","pt"): "",
        ("sv","pl"): "",
        ("sv","ja"): "",
        ("sv","ko"): "",
        ("sv","vi"): "",

        ("pl","en"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zlw-en",
        ("pl","de"): "opus_mt_models/Helsinki-NLP__opus-mt-pl-de",
        ("pl","fr"): "opus_mt_models/Helsinki-NLP__opus-mt-pl-fr",
        ("pl","es"): "opus_mt_models/Helsinki-NLP__opus-mt-pl-es",
        ("pl","ru"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zlw-zle",
        ("pl","uk"): "opus_mt_models/Helsinki-NLP__opus-mt-tc-big-zlw-zle",
        ("pl","sv"): "opus_mt_models/Helsinki-NLP__opus-mt-pl-sv",
        ("pl","ja"): "", #back track wie?
        ("pl","it"): "",
        ("pl","pt"): "",
        ("pl","nl"): "",
        ("pl","tr"): "",
        ("pl","zh"): "",
        ("pl","ko"): "",
        ("pl","vi"): "",
    }
    
    return langs_to_model.get((src, tgt), "")

def get_token(tgt):
    lang_to_token = {
        "en": ">>eng<<",
        "de": ">>deu<<",
        "fr": ">>fra<<",
        "zh": ">>zho_Hans<<",  # /: >>cmn_Hans<<
        "ru": ">>rus<<",
        "uk": ">>ukr<<",
        "es": ">>spa<<",
        "pt": ">>por<<",
        "tr": ">>tur<<",
        "it": ">>ita<<",
        "ja": ">>jpn<<",
        "sv": ">>swe<<",
        "pl": ">>pol<<",
        "ko": ">>kor<<",
        "nl": ">>nld<<",
        "vi": ">>vie<<",
    }
    return lang_to_token[tgt]

def needs_prefix(model_name):
    parts = model_name.split("-")
    
    tgt = parts[-1]
    
    if len(tgt) == 2:
        return False
    
    return True

# @app.post("/preload_model")
# async def preload_model(background_tasks: BackgroundTasks, source_lang: str = Form(...), target_lang: str = Form(...)):
#     model_name = get_opus_model_name(source_lang, target_lang)

#     if model_name:
#         background_tasks.add_task(get_model, model_name)

#     return {"status": "loading"}

@app.get("/")
async def root():
    return {"message": "TalkBridge WebSocket Server", "status": "running"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80, log_level="info")
    #Local: ws://localhost:80/ws/translate")
    #Network: ws://192.168.178.74:80/ws/translate")
    #Emulator: ws://10.0.2.2:80/ws/translate")