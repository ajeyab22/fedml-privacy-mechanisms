import fedml
from fedml import FedMLRunner

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    print("Fuck 1", args.client_num_in_total)
    fedml_runner = FedMLRunner(args, device, dataset, model)
    print("Fuck 2", args.client_num_in_total)
    fedml_runner.run()
    print("Fuck 3", args.client_num_in_total)
