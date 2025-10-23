import os
import json
import tempfile
import traceback
import boto3
import requests


TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'storage.yandexcloud.net')
S3_REGION = os.environ.get('S3_REGION', 'ru-central1')
YC_STORAGE_KEY_ID = os.environ.get('YC_STORAGE_KEY_ID')
YC_STORAGE_SECRET_KEY = os.environ.get('YC_STORAGE_SECRET_KEY')


def send_tg(chat_id: int, text: str):
    if not TELEGRAM_BOT_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    except Exception:
        pass


def handler(event, context=None):
    # YMQ event shape: either {'messages': [...]} или SQS-compatible {'Records': [...]}.
    try:
        records = event.get('messages') or event.get('Records') or []
    except Exception:
        records = []

    for rec in records:
        try:
            body = rec.get('details', {}).get('message', {}).get('body') if 'details' in rec else rec.get('body')
            payload = json.loads(body)

            chat_id = payload['chat_id']
            job_id = payload['job_id']
            s3_bucket = payload['s3_bucket']
            s3_key = payload['s3_key']
            dependent_vars = payload['dependent_vars']
            independent_vars = payload['independent_vars']
            subgroups = payload.get('subgroups')
            analysis_type = payload.get('analysis_type', 'total')
            min_sample_size = int(payload.get('min_sample_size', 100))

            send_tg(chat_id, f"🚀 Старт расчёта (job {job_id})… Загружаю данные…")

            s3 = boto3.client(
                's3',
                endpoint_url=f"https://{S3_ENDPOINT}",
                region_name=S3_REGION,
                aws_access_key_id=YC_STORAGE_KEY_ID,
                aws_secret_access_key=YC_STORAGE_SECRET_KEY,
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as tmpf:
                s3.download_fileobj(s3_bucket, s3_key, tmpf)
                local_path = tmpf.name

            send_tg(chat_id, "📊 Данные загружены, запускаю полный анализ…")

            from johnson_weights import calculate_johnson_weights
            results_path = calculate_johnson_weights(
                input_file=local_path,
                dependent_vars=dependent_vars,
                independent_vars=independent_vars,
                subgroups=subgroups,
                min_sample_size=min_sample_size,
                output_dir="results"
            )

            if not results_path:
                send_tg(chat_id, "❌ Ошибка: результаты не были сгенерированы.")
                continue

            base_key = f"results/{job_id}/"
            s3.upload_file(results_path, s3_bucket, base_key + os.path.basename(results_path))
            csv_path = results_path.replace('.xlsx', '.csv')
            if os.path.exists(csv_path):
                s3.upload_file(csv_path, s3_bucket, base_key + os.path.basename(csv_path))

            try:
                url_doc = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
                with open(results_path, 'rb') as fxl:
                    requests.post(url_doc, data={"chat_id": chat_id}, files={"document": fxl}, timeout=60)
                if os.path.exists(csv_path):
                    with open(csv_path, 'rb') as fcsv:
                        requests.post(url_doc, data={"chat_id": chat_id}, files={"document": fcsv}, timeout=60)
            except Exception:
                pass

            send_tg(chat_id, "✅ Расчёт завершён. Файлы также сохранены в хранилище.")

            status = {
                "job_id": job_id,
                "state": "success",
            }
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as st:
                st.write(json.dumps(status).encode('utf-8'))
                s3.upload_file(st.name, s3_bucket, f"results/{job_id}/status.json")

        except Exception as e:
            try:
                chat_id = payload.get('chat_id') if 'payload' in locals() else None
                if chat_id:
                    send_tg(chat_id, f"❌ Ошибка обработки задачи: {e}")
            except Exception:
                pass
            traceback.print_exc()

    return {"status": "ok"}


