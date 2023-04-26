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
    st = time.time()
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
    et = time.time()
    agg_time=et-st
    print("Total time for running:"+str(agg_time)+" seconds")
    
