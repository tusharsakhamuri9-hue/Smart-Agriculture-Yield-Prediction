from predict import predict_yield

result = predict_yield(
    area=100,
    crop="Rice",
    state="Andhra Pradesh",
    season="Kharif",
    district="ANANTAPUR",
    crop_year=2015
)

print("Predicted Yield:", result)