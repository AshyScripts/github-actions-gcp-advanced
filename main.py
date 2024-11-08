import logging
import subprocess

def retrain_model(request):
    # Basic logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info('Starting model retraining process')
        
        # Run data processing
        data_process = subprocess.run(
            ['python', 'data_processing.py'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if data_process.returncode != 0:
            return (f"Data processing failed: {data_process.stderr}", 500)
            
        # Run model training
        train_result = subprocess.run(
            ['python', 'train_model.py'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if train_result.returncode == 0:
            return ('Model retrained successfully.', 200)
        else:
            return (f'Error in retraining: {train_result.stderr}', 500)
            
    except Exception as e:
        return (f'Error: {str(e)}', 500)