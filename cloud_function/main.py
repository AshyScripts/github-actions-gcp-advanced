# main.py

def retrain_model(request):
    import subprocess

    result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
    if result.returncode == 0:
        return ('Model retrained successfully.', 200)
    else:
        return (f'Error in retraining: {result.stderr}', 500)
