import json

def map_label(map_path='cat_to_name.json'):
    '''
        Return a dictionary mapping the integer encoded categories
        to the actual names of the flowers
    '''
    with open(map_path, 'r') as f:
        cat_to_name = json.load(f)

    #view the different flower categories
    #print(cat_to_name)
    return cat_to_name
