import traceback
try:
    from fer import FER
    print("fer import successful")
except Exception:
    traceback.print_exc()
