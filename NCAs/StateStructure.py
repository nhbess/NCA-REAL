class StateStructure:
    def __init__(self, estimation_dim:int, constant_dim:int, sensor_dim:int, hidden_dim:int):
        self.estimation_channels = slice(0, estimation_dim)
        self.constant_channels = slice(estimation_dim, estimation_dim + constant_dim)
        self.sensor_channels = slice(estimation_dim + constant_dim, estimation_dim + constant_dim + sensor_dim)
        self.hidden_channels = slice(estimation_dim + constant_dim + sensor_dim, estimation_dim + constant_dim + sensor_dim + hidden_dim)

        self.state_dim = estimation_dim + constant_dim + sensor_dim + hidden_dim
        self.estimation_dim = estimation_dim
        self.constant_dim = constant_dim
        self.sensor_dim = sensor_dim
        self.hidden_dim = hidden_dim

        self.out_dimension = estimation_dim + hidden_dim
        self.out_estimation_channels = slice(0, estimation_dim)
        self.out_hidden_channels = slice(estimation_dim, estimation_dim + hidden_dim)
        
if __name__ == '__main__':
    pass