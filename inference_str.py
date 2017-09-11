import tensorflow as tf

from nmt import inference
from nmt import nmt

if __name__ == "__main__":

    ckpt = None
    out_dir = '/Users/taatoal3/Desktop/best_bleu'
    hparams = nmt.create_or_load_hparams(out_dir, {}, {})
    jobid = 0
    num_workers = 1

    if not ckpt:
        ckpt = tf.train.latest_checkpoint(out_dir)
    result = inference.inference_str(ckpt, "ceci n'est pas une pipe", hparams, num_workers, jobid)
    print(result)
    # calls single_worker_inference
    # single_worker_inference uses load_data(input file, hparams)
    # returns a list of lines
