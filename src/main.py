import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    string = 'dir_data'
    print(getattr(args, string))
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            model_t = model.Model(args, checkpoint)
            model_s = model.Model(args, checkpoint, student=True)
            model = [model_t, model_s]
            loss_ = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, model, loss_, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
