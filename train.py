import config
import trainer_step


def main():
    cfg = config.Config(filenamequeue="./dataset/layout_1205.tfrecords")
    t = trainer_step.Trainer(cfg)
    t.fit()


if __name__ == "__main__":
    main()
