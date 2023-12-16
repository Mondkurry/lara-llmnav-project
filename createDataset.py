import os
import json
import random
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from tqdm import tqdm
from PIL import Image

class RandomPatch:
    def __init__(self, city='San Francisco, California, USA', max_iter=100):
        self.city = city
        self.max_iter = max_iter
        self.patch = None
        self.street_id_mapping = {} 
        self.errors = []
        self.image_size = 550  # Size of the image to be saved
        self.initialize_patch()
        self.start_node = None
        self.end_node = None
        self._select_start_end_nodes()
        
    def initialize_patch(self):
        city_center = ox.geocode(self.city)
        success = False
        scaling_factor = 3  # Adjust as needed

        for _ in range(self.max_iter):
            try:
                # Generate a random point within approximately 1 km of the city center
                random_point = (
                    city_center[0] + scaling_factor * random.uniform(-0.009, 0.009),
                    city_center[1] + scaling_factor * random.uniform(-0.012, 0.012)
                )

                # Attempt to create a graph from the point
                self.patch = ox.graph_from_point(random_point, dist=200, dist_type='bbox', network_type='drive')

                # Check if the graph is empty
                if self.patch is None or self.patch.number_of_nodes() == 0:
                    self.errors.append("\033[91mError:\033[0m Generated graph is empty. Retrying...")
                    continue

                # Check if the graph has at least 5 nodes
                if self.patch.number_of_nodes() >= 5:
                    if self._select_start_end_nodes():
                        self._generate_street_ids()
                        success = True
                        break
            except Exception as e:
                self.errors.append(f"\033[91mError:\033[0m Error during graph generation: {e}. Retrying with a new point...")
                continue

        if not success:
            self.errors.append("\033[91mError:\033[0m Failed to find a suitable patch with a connecting path in the given number of iterations.")

    def _generate_street_ids(self):
        streets = set()
        for _, _, data in self.patch.edges(data=True):
            if 'name' in data:
                streets.add(data['name'])
        for i, street in enumerate(streets):
            self.street_id_mapping[street] = f"S{i+1}"

    def _select_start_end_nodes(self):
        nodes = list(self.patch.nodes)
        random.shuffle(nodes)

        for start_node in nodes:
            for end_node in nodes:
                if start_node != end_node and nx.has_path(self.patch, start_node, end_node):
                    self.start_node = start_node
                    self.end_node = end_node
                    return True
        return False
    
    def calculate_bearing(self, point1, point2):
        """
        Calculate the bearing between two points.
        """
        lat1, lon1 = point1
        lat2, lon2 = point2
        delta_lon = np.radians(lon2 - lon1)
        lat1, lat2 = np.radians(lat1), np.radians(lat2)

        x = np.sin(delta_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

        bearing = np.arctan2(x, y)
        bearing = np.degrees(bearing)
        bearing = (bearing + 360) % 360

        return bearing

    def calculate_turn_direction(self, bearing1, bearing2):
        """
        Determine the turn direction from one bearing to another.
        """
        delta = (bearing2 - bearing1 + 360) % 360
        if delta > 180:
            return 'left'
        else:
            return 'right'

    def get_turn_by_turn_instructions(self):
        """
        Generate turn-by-turn instructions for the shortest path.
        """
        if not nx.has_path(self.patch, self.start_node, self.end_node):
            return "No path found."

        path = nx.shortest_path(self.patch, self.start_node, self.end_node)
        instructions = []
        prev_street_name = None
        prev_bearing = None

        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            edge_data = self.patch.get_edge_data(start_node, end_node)[0]
            
            street_name = edge_data.get('name', 'Unnamed Road')
            street_id = self.street_id_mapping.get(street_name, 'Unnamed Road')  # Use street ID
            
            # Get node coordinates
            start_coords = (self.patch.nodes[start_node]['y'], self.patch.nodes[start_node]['x'])
            end_coords = (self.patch.nodes[end_node]['y'], self.patch.nodes[end_node]['x'])

            # Calculate bearing
            bearing = self.calculate_bearing(start_coords, end_coords)

            if i == 0:
                instructions.append(f"straight from Starting Node to {street_id}")
            else:
                if prev_street_name != street_id:
                    turn_direction = 'straight' if prev_bearing is None else self.calculate_turn_direction(prev_bearing, bearing)
                    instructions.append(f"{turn_direction} from {prev_street_name} to {street_id}")

            prev_street_name = street_id
            prev_bearing = bearing

        if prev_street_name:
            instructions.append(f"straight from {prev_street_name} to Ending Node")

        return instructions
    
    def plot_patch(self, show_graph = True, plot_path=False, save_fig=False, fig_name=None):
        if self.patch and self.start_node and self.end_node:
            # Plot the graph
            fig, ax = ox.plot_graph(self.patch, show=False, close=False)
            
            # Plot street names

            for _, edge in ox.graph_to_gdfs(self.patch, nodes=False).fillna('').iterrows():
                if isinstance(edge['geometry'], LineString):
                    x, y = edge['geometry'].xy
                    c = edge['geometry'].centroid
                    
                    street_name = edge['name']
                    text = self.street_id_mapping.get(street_name, '999')  # Use street ID

                    # Calculate rotation angle
                    dx = x[-1] - x[0]
                    dy = y[-1] - y[0]
                    angle = np.degrees(np.arctan2(dy, dx))

                    # Calculate offset for the text (perpendicular to the road)
                    offset_mag = 0.0001  # Adjust the magnitude of the offset as needed
                    perp_angle = np.radians(angle + 90)  # Perpendicular angle
                    offset_x = np.cos(perp_angle) * offset_mag
                    offset_y = np.sin(perp_angle) * offset_mag

                    # Apply offset to the centroid coordinates
                    text_x = c.x + offset_x
                    text_y = c.y + offset_y
                    
                    ax.annotate(text, (text_x, text_y), rotation=angle, ha='center', va='center', color='w', fontsize=12)

            # Check if a path exists
            if nx.has_path(self.patch, self.start_node, self.end_node):
                # Find the shortest path
                shortest_path = nx.shortest_path(self.patch, self.start_node, self.end_node)

                # Extract the x and y coordinates for each node in the path
                x_coords = [self.patch.nodes[node]['x'] for node in shortest_path]
                y_coords = [self.patch.nodes[node]['y'] for node in shortest_path]

                # Plot the path
                if plot_path:
                    ax.plot(x_coords, y_coords, color='green', linewidth=6, alpha=0.5)

                # Plot start and end nodes
                start_x, start_y = self.patch.nodes[self.start_node]['x'], self.patch.nodes[self.start_node]['y']
                end_x, end_y = self.patch.nodes[self.end_node]['x'], self.patch.nodes[self.end_node]['y']
                ax.scatter(start_x, start_y, color='green', s=100)
                ax.scatter(end_x, end_y, color='red', s=100)

            if save_fig:
                if fig_name:
                    plt.savefig(fig_name, format='png', dpi=80, pad_inches=0)
                    plt.close()  # Close the figure
                    
                    # Open the image and crop it to the desired size
                    img = Image.open(fig_name)
                    width, height = img.size   # Get dimensions
                    
                    # Calculate the area to crop
                    left = (width - self.image_size) / 2
                    top = (height - self.image_size) / 2
                    right = (width + self.image_size) / 2
                    bottom = (height + self.image_size) / 2

                    # Crop the center of the image
                    img = img.crop((left, top, right, bottom))
                    img.save(fig_name)  # Save the cropped image
                    img.close()  # Close the image file
                else:
                    self.errors.append("\033[91mError:\033[0m No figure name provided.")

                if show_graph:
                    plt.show()
                else:
                    plt.close()  # Close the figure when not showing it
            else:
                self.errors.append("\033[91mError:\033[0m No path found between the start and end nodes.")
        else:
            self.errors.append("\033[91mError:\033[0m No patch or start/end nodes have been initialized.")

class createDataset():
    def __init__(self, save_path, locations, num_datapoints_per_city=3):
        self.locations = locations
        self.num_datapoints_per_city = num_datapoints_per_city
        self.save_path = save_path

    def _create_directories(self, city):
        city_folder = os.path.join(self.save_path, city)
        images_folder = os.path.join(city_folder, 'images')
        directions_folder = os.path.join(city_folder, 'directions')

        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(directions_folder, exist_ok=True)

        return city_folder, images_folder, directions_folder

    def _append_directions_to_json(self, single_direction, json_path):
        # Read existing data
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
        else:
            data = []

        # Append the new direction
        data.append(single_direction)

        # Write back to the file
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def generateDataset(self):
        for city in self.locations:
            print(f"\nGenerating dataset for \033[95m{city}\033[0m ...\n")
            city_folder, images_folder, directions_folder = self._create_directories(city)
            directions_json_path = os.path.join(directions_folder, f'{city}_directions.json')

            with tqdm(total=self.num_datapoints_per_city) as pbar:
                for i in range(self.num_datapoints_per_city):
                    try:
                        random_patch = RandomPatch(city=city)
                        image_name = f'{city}_Start_{random_patch.start_node}_End_{random_patch.end_node}.png'
                        fig_name = os.path.join(images_folder, image_name)

                        random_patch.plot_patch(show_graph=False, plot_path=False, save_fig=True, fig_name=fig_name)

                        directions = random_patch.get_turn_by_turn_instructions()
                        direction_data = {
                            'image_name': image_name,
                            'directions': directions
                        }

                        self._append_directions_to_json(direction_data, directions_json_path)
                        
                        # Log errors, if any, from the random_patch
                        for error in random_patch.errors:
                            pbar.write(error)
                        pbar.update(1)
                        
                    except Exception as e:
                        pbar.write(f"\033[91mError:\033[0m Error occurred for {city} at datapoint {i}: {e}. Skipping this datapoint.")


###################################### CREATING THE DATASET ######################################


dataset = createDataset(save_path='data_test', 
                        locations=[ 
                                    'San Francisco, California, USA', 
                                    'New York City, New York, USA',
                                    'Los Angeles, California, USA',
                                    'Austin, Texas, USA',
                                    'Chicago, Illinois, USA',
                                    'Boston, Massachusetts, USA',
                                    'San Diego, California, USA',
                                    'San Jose, California, USA',
                                    'Miami, Florida, USA',
                                    ], 
                        num_datapoints_per_city=1000)

dataset.generateDataset()


###################################### DATASET INFORMATION ######################################

# City: Num Images
# San Francisco: 995 images
# New York: 984 images
# San Jose: 986 images
# San Diego: 973 images
# Miami: 971 images
# Los Angeles: 993 images
# Chicago: 979 images
# Boston: 127 images
# Austin: 992 images