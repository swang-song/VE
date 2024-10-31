
import os
import sys
import logging

# Redirect stdout and stderr to logging
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''
    
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line)
    
    def flush(self):
        pass

def model_log(args):
    msgModel = 'k{}_p{}_fn{}'.format(
        args.VE_len,
        args.ParamRatio,
        args.usePreprocess,
    )
    
    setting = '{}_{}_{}'.format(
                args.model_id,
                args.model,
                msgModel,
                )
    return setting


def log_config(logger_file, args):
    dataset = args.model_id.split('_')[0]
    folder_path = './results/terminal/' + args.model + '/' + dataset + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    logger_file = folder_path + logger_file
    with open(logger_file, "w") as file:
        file.truncate(0)
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Adjust level as needed (e.g., DEBUG, WARNING)
        format='%(message)s',
        handlers=[
            logging.FileHandler(logger_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
