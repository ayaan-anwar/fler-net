class HyperParameters:
    def __init__(self):
        self.activation = 'relu'
        self.layer1 = 256
        self.dropout1 = 0.3
        self.layer2 = 64
        self.dropout2 = 0.3
        self.layer3 = 16
        self.dropout3 = 0.2
        self.output_layer = 1
        self.output_activation = 'sigmoid'
        self.base_loss = 0.01
        self.fed_loss_client = 0.1
        self.fed_loss_server = 1.0
        self.epochs = 10000
        self.batch_size = 256
        self.fed_rounds = 20000
        self.epochs_per_round = 5
        self.shuffle = 1000
