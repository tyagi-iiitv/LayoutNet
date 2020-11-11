import os
class Config(object):
    def __init__(self, **kwargs):
        # configuration for building the network
        self.data_dir = os.path.join('..','data')
        self.y_dim = kwargs.get("y_dim", 6)
        self.tr_dim = kwargs.get("tr_dim", 7)
        self.ir_dim = kwargs.get("ir_dim", 10)
        self.latent_dim = kwargs.get("latent_dim", 128)
        self.z_dim = kwargs.get("z_dim", 128)
        self.batch_size = kwargs.get("batch_size", 128)
        self.lr = kwargs.get("lr", 0.0002)
        self.beta1 = kwargs.get("beta1", 0.5)
        # configuration for the supervisor
        self.logdir = kwargs.get("logdir", "./log")
        self.sampledir = kwargs.get("sampledir", os.path.join(self.data_dir, "example"))
        self.max_steps = kwargs.get("max_steps", 30000)
        self.sample_every_n_steps = kwargs.get("sample_every_n_steps", 100)
        self.summary_every_n_steps = kwargs.get("summary_every_n_steps", 1)
        self.save_model_secs = kwargs.get("save_model_secs", 120)
        self.checkpoint_basename = kwargs.get("checkpoint_basename",
                                              "layout")
        self.filenamequeue = kwargs.get("filenamequeue", os.path.join(self.data_dir, "dataset","layout_1205.tfrecords"))
        self.min_after_dequeue = kwargs.get("min_after_dequeue", 5000)
        self.num_threads = kwargs.get("num_threads", 4)
