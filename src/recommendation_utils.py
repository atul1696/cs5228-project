import numpy as np

def get_recommendation_weights(row, X, feature_weightage):
    similarity_dict = {}
    ## TODO : The comparisons below do not take into account what happens if the input is NaN
    ## For example, if the input floor level is NaN, we don't want to give higher importance to other
    ## properties which also have floor level NaN. Instead, we just want to ignore that feature
    ## (maybe just by giving score '0' for all data points if the input itself doesn't contain that information)

    ## Are they in the same subzone?
    similarity_dict['subzone'] = np.array(X['subzone']==row['subzone'])

    ## Are they in the same planning area? (Should probably be less importance than subzone)
    similarity_dict['planning_area'] = 0.5*np.array(X['planning_area']==row['planning_area'])
    ## TODO : Maybe we can remove both subzone and planning area and replace it all with lat, lng distance

    ## Are they in the same price bracket?
    print(np.max(X['price']))
    print(np.min(X['price']))
    print(X['price'].isnull().values.any())
    hist, bin_edges = np.histogram(X['price'], bins=100)
    similarity_dict['price'] = np.digitize(X['price'], bin_edges) == np.digitize(row['price'], bin_edges)
    ## TODO : Convert price difference from categorical to continuous

    ## Do they have the same property type?
    similarity_dict['property_type'] = np.array(X['property_type']==row['property_type'])

    ## Do they have the same number of beds?
    similarity_dict['num_beds'] = np.array(X['num_beds']==row['num_beds'])
    ## TODO : We can convert this to continous as well.
    ## But should we? I think if someone is looking for say 3 bedroom apartments,
    ## it's not likely they'd be open to 2 bedroom or 4 bedroom units. That part is usually not flexible.

    ## Are they in the same size bracket?
    hist, bin_edges = np.histogram(X['size_sqft'], bins=100)
    similarity_dict['size_sqft'] = np.digitize(X['size_sqft'], bin_edges) == np.digitize(row['size_sqft'], bin_edges)
    ## TODO : Convert size difference from categorical to continuous
    ## TODO : Add another feature for 'price per square foot'

    ## Do they have the same floor level?
    similarity_dict['floor_level'] = np.array(X['floor_level']==row['floor_level'])

    ## Do they have the same level of furnishing?
    similarity_dict['furnishing'] = np.array(X['furnishing']==row['furnishing'])

    ## How much time is left in the tenure?
    ## TODO : Need data preprocessing first. Then we can simply subtract difference between built year and current year from tenure

    weights = np.zeros(len(X))
    for ele in similarity_dict:
        weights = weights + feature_weightage[ele]*similarity_dict[ele]

    return weights
