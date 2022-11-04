import numpy as np

def get_recommendation_weights(row, X):
    similarity_dict = {}
    ## Are they in the same subzone?
    similarity_dict['subzone'] = np.array(X['subzone']==row['subzone'])

    ## Are they in the same planning area? (Should probably be less importance than subzone)
    similarity_dict['planning_area'] = 0.5*np.array(X['planning_area']==row['planning_area'])

    ## Are they in the same price bracket?
    hist, bin_edges = np.histogram(X['price'], bins=100)
    similarity_dict['price'] = np.digitize(X['price'], bin_edges) == np.digitize(row['price'], bin_edges)

    ## Other Things we can try
    ## Property Type
    ## Time Left in Tenure
    ## Num Beds
    ## Size Sqft
    ## Floor Level
    ## Furnishing

    weights = np.zeros(len(X))
    for ele in similarity_dict:
        weights = weights + similarity_dict[ele]

    return weights
