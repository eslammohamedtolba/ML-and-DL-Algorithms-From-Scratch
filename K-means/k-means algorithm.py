# Load dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(21)


# Create Cluster class to represent each cluster
class Cluster:
    def __init__(self):
        self.centroid = None  # Centroid of the cluster
        self.l = []  # List to store instances of the cluster

    # Method to set the centroid of the cluster
    def set_centroid(self, new_centroid):
        self.centroid = new_centroid

    # Method to add an instance to the cluster
    def set_instance(self, instance):
        self.l.append(instance)

    # Method to recenter the cluster (calculate new centroid based on mean of all instances)
    def recenter(self):
        # Convert the list of instances into a numpy array for easier manipulation
        instances_array = np.array(self.l)
        
        # Calculate the mean of all instances (new centroid)
        new_centroid = np.mean(instances_array, axis=0)

        # Update the cluster's centroid
        self.set_centroid(new_centroid)

        # Clear the list of instances as we've updated the centroid
        self.l = []


# Create kmeans class to implement the K-Means algorithm
class kmeans:
    def __init__(self, k = 3, max_iter = 1000, method = 'euclidean'):
        self.k = k  # Number of clusters
        self.clusters = []  # List of clusters
        self.max_iter = max_iter  # Maximum number of iterations
        # Set method for calculating distance (Euclidean or Manhattan)
        self.method = 'euclidean' if method == 'euclidean' else 'manhattan'

    # Method to initialize centroids by picking random instances from the dataset
    def __initialize_centroids(self, data):
        # Randomly pick indices from the dataset to set as initial centroids
        indices = np.random.choice(data.shape[0], self.k, replace = False)
        instances = data.iloc[indices,:].values  # Get the values of the randomly chosen instances
        for i in range(self.k):
            self.clusters.append(Cluster())  # Create a new cluster
            self.clusters[i].set_centroid(instances[i])  # Set the centroid for the cluster

    # Method to calculate Euclidean distance between two points
    def euclidean_method(self, v1, v2):
        euclidean_distance = np.sqrt(np.sum((v1 - v2) ** 2))  # Euclidean distance formula
        return euclidean_distance

    # Method to calculate Manhattan distance between two points
    def manhattan_method(self, v1, v2):
        manhattan_distance = np.sum(np.abs(v1 - v2))  # Manhattan distance formula
        return manhattan_distance
    
    # Method to compute the distance from an instance to each centroid and return the closest centroid
    def ___compute_distance(self, instance):
        instance_results = []
        for cluster in self.clusters:
            centroid = cluster.centroid  # Get the current centroid
            if self.method == 'euclidean':  # Use Euclidean method if specified
                instance_results.append(self.euclidean_method(centroid, instance))
            else:  # Use Manhattan method if specified
                instance_results.append(self.manhattan_method(centroid, instance))
        
        # Return the index of the closest centroid
        cluster_index = np.argmin(instance_results)  # Find the minimum distance (closest centroid)
        return cluster_index

    # Method to assign each instance to the nearest cluster
    def __Construct_Clusters(self, data):
        for i in range(data.shape[0]):
            instance = data.iloc[i,:].values  # Get the current instance as a numpy array
            cluster_index = self.___compute_distance(instance)  # Find the closest centroid
            self.clusters[cluster_index].set_instance(instance)  # Add the instance to the cluster

    # Method to recenter all clusters (update centroids)
    def __compute_centroids(self):
        for i in range(len(self.clusters)):
            self.clusters[i].recenter()  # Recompute the centroid for each cluster

    # Main fitting method to train the k-means model
    def fit(self, data):
        self.__initialize_centroids(data)  # Initialize the centroids
        for _ in range(self.max_iter):  # Iterate for the given number of times
            self.__Construct_Clusters(data)  # Assign instances to the closest clusters
            self.__compute_centroids()  # Update centroids based on assigned instances
    
    # Method to predict the cluster for a new instance
    def predict(self, instance):
        cluster_number = self.___compute_distance(instance)  # Find the closest cluster
        return cluster_number  # Return the cluster index


# Read and visualize the dataset
data = pd.read_csv('synthetic_dataset_three_clusters.csv')  # Load data from CSV file

# Create an instance of the kmeans class and fit it to the data
kmeans = kmeans()  # Instantiate the kmeans object
kmeans.fit(data)  # Fit the model to the data

# Get the clusters after fitting
clusters = kmeans.clusters

# Plot the dataset and show the centroids for each cluster
plt.figure(figsize = (7,7))  # Create a figure with a size of 7x7 inches
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color = 'b', s = 10)  # Scatter plot of the data points
plt.scatter([x.centroid[0] for x in clusters], [x.centroid[1] for x in clusters], color = ['r','y','g'], s = 50)  # Scatter plot of centroids
plt.title('Synthetic Dataset with Three Clusters')  # Set plot title
plt.xlabel("X1")  # Set x-axis label
plt.ylabel("X2")  # Set y-axis label
plt.legend()  # Add legend
plt.show()  # Display the plot

# Test the model by predicting cluster for new instances
instances = np.array([[0, 0], [4, 8], [7, 3]])  # Define new test instances
for instance in instances:
    print(kmeans.predict(instance))  # Predict and print the cluster index for each instance
