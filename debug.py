def predict_spam(request):
    if request.method == "POST":
        import json
        data = json.loads(request.body)
        text = data.get("text", "")

        print(f"Received text: {text}")  # Debug print

        transformed_text = vectorizer.transform([text])
        print(f"Transformed Text: {transformed_text}")

        try:
            spam_probability = lr_model.predict_proba(transformed_text)[0][1]
            print(f"Spam Probability: {spam_probability}")  # Debug print
            return JsonResponse({"spam_probability": round(float(spam_probability), 2)})
        except Exception as e:
            print(f"Error during prediction: {str(e)}")  # Debug print
            return JsonResponse({"error": "Prediction failed"}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
