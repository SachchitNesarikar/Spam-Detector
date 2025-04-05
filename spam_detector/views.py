from ml_algos.models import predict_spam
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def spam_detection_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get("text", "")
            model_choice = data.get("model_choice", "naive_bayes")
            result = predict_spam(text, model_choice)

            if "error" in result:
                return JsonResponse(result, status=400)

            response_data = {
                "input": result["input"],
                "cleaned": result["cleaned"],
                "model": result["model"],
                "is_spam": bool(result["is_spam"]),
                "spam_probability_percent": float(result["spam_probability"]) * 100
            }

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({"error": "Server error"}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)