import numpy as np

def make_feature_activation(num_samples, mean=-3.5, stdev=1):
    # mean of -3.5 corresponds to being active 300/1000000 times, like the average Anthropic feature
    return np.maximum(np.zeros((num_samples)),np.random.normal(loc=mean, scale=stdev, size=(num_samples)))

def make_neuron_activation(num_samples, mean=0, stdev=1):
    return np.maximum(0,np.random.normal(loc=mean, scale=stdev, size=(num_samples)))

def correlation_of_feature_with_many_neurons(num_samples, num_neurons):
    max_correlation=-1
    feature_activation=make_feature_activation(num_samples)
    for neuron_number in range(num_neurons):
        neuron_activation=make_neuron_activation(num_samples)
        this_correlation=np.corrcoef(np.stack((feature_activation, neuron_activation)))[0,1]
        max_correlation=max(max_correlation,this_correlation)
    return max_correlation
    

if __name__=="__main__":
    num_samples=int(1e5)
    num_neurons=int(1e3) #should 2.4e5, but that is slow to run
    for _ in range(100):
        print(correlation_of_feature_with_many_neurons(num_samples, num_neurons))
    