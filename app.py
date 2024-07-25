## ----------------------- KUTUBXONALAR -------------------------------##
import streamlit as st
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
##-------------------------------------------------------------------------
st.set_page_config(page_title="Filigran Distribution",
                   page_icon="./images/logo.png",
                   layout="wide")
# Add image to sidebar
st.logo('./images/logo.png')
# st.sidebar.image("logo2.png", width=250)

### ---------------------- Bacground style -----------------------------###
# original_title = "<h1 style='font-family: serif; color:white; font-size: 20px;'>Link data yordamida eng optimal yo'l xaritasi✨</h1>"
# st.markdown(original_title, unsafe_allow_html=True)

# Set the background image for the entire Streamlit app, including the sidebar
background_image = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://media.istockphoto.com/id/1226478926/photo/green-background-3d-render.jpg?s=612x612&w=0&k=20&c=_2N-T1myybE8kLkNCpyzhiLXcVlhlezNEoCSe9Nsgm8=");
    background-size: cover;  /* This sets the size to cover 100% of the container */
    background-position: center;  
    background-repeat: no-repeat;
    background-attachment: fixed; /* Ensures the background image stays fixed while scrolling */
}

[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.2) !important; /* Makes the sidebar background more transparent */
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0.2) !important;
}

[data-testid="stToolbar"] {
    right: 2rem;
}

</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

# Style the text input
input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  /* Changes the text color inside the input box */
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
</style>
"""

st.markdown(input_style, unsafe_allow_html=True)
### ---------------------  Clean the data  ---------------------------------###

# ____________data upload ____________
init_streamlit_comm()

@st.cache_resource
def get_pyg_html(df: pd.DataFrame) -> str:
    # When you need to publish your application, you need set `debug=False`,prevent other users to write your config file.
    # If you want to use feature of saving chart config, set `debug=True`
    html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
    return html

@st.cache_data
def get_df(data_file) -> pd.DataFrame:
    return pd.read_excel(data_file)
 
df = None
##---------------------------------- SIDEBAR ------------------------------------------------------------
with st.sidebar:
    uploaded_files = st.sidebar.file_uploader("Excel faylni yuklang", type=['xlsx'])
    if uploaded_files is not None:
        if uploaded_files.name != 'Mappe1.xlsx':
            st.markdown('<div class="custom-error">SIZ XATO FORMATDAGI DATANI YUKLAYAPSIZ 🚨</div>', unsafe_allow_html=True)
        else:
            # Load and process the data from the uploaded file
            # Example of how you might read and process the file
            df = pd.read_excel(uploaded_files)
    st.write("")
    st.sidebar.markdown('<h3 style="color:white;">Qaysi Mashinalar mavjud ?</h3>', unsafe_allow_html=True)
    options = st.sidebar.multiselect("", ['Vehicle 1', 'Vehicle 2', 'Vehicle 3', 'Vehicle 4', 'Vehicle 5',
        'Vehicle 6', 'Vehicle 7'])

    button = st.sidebar.button("Transportlarni tasdiqlash")
    
#---------------------------------------------------------------------------------------------------------------

if button:
    if len(options) >= 2:
        if uploaded_files is not None:
            df = get_df(uploaded_files)
            orders = df
            # st.write(orders)
            

            # Extract the headers from the first three rows for multi-level indexing
            headers = orders.iloc[0:4]

            # Set the first four columns with appropriate headers based on row 2 and the rest with combined headers
            new_columns = headers.iloc[3, :4].tolist() + headers.apply(lambda x: f'{x[0]} {x[1]} {x[2]} {x[3]}', axis=0)[4:].tolist()

            # Rename the columns in the dataframe
            orders.columns = new_columns
            orders = orders.drop(index=[0, 1, 2, 3])  # Remove the header rows

            # Reset the index after dropping header rows
            orders.reset_index(drop=True, inplace=True)

            # Melt the dataframe to long format
            long_format_df = pd.melt(orders, 
                                    id_vars=["Customer ID", "Customer name", "Customer location", "District"], 
                                    var_name="Product_Info", 
                                    value_name="Quantity")

            # Split the 'Product_Info' into 'Product Category', 'Volume name', and 'Assortiment'
            long_format_df[['Product Category', 'Volume name', 'Assortiment', 'Product ID']] = long_format_df['Product_Info'].str.split(' ', expand=True)

            # Remove the 'Product_Info' column as it is no longer needed
            long_format_df.drop(columns=['Product_Info'], inplace=True)

            # Replace NaNs in 'Quantity' with 0
            long_format_df['Quantity'].fillna(0, inplace=True)

            # Convert 'Quantity' to integer
            long_format_df['Quantity'] = long_format_df['Quantity'].astype(int)

            # Reorder the columns to match specified format
            long_format_df = long_format_df[['Customer ID', 'Customer name', 'Customer location', 'District', 
                                            'Product Category', 'Volume name', 'Assortiment', 'Product ID', 'Quantity']]
            # st.write(long_format_df)
            ### -------------------------------------------------------------------------------------------------------------
            # Load the 'Orders' sheet from the Excel file without using the first rows as column names
            products_df = pd.read_excel('data/products_df.xlsx')
            vehicles_ = pd.read_excel('data/vehicles_df_C30.xlsx')
            vehicles_df = vehicles_[vehicles_['Vehicle name'].isin(options)]
            warehouse_df = pd.read_excel('data/warehouse_df.xlsx')
            customers_df = pd.read_excel('data/customers_df_C30.xlsx')

            # Merge the products_df with long_format_df on Product ID to calculate total volume required per order
            order_details_df = pd.merge(long_format_df, products_df[['Product ID', 'Volume cm3']], on='Product ID', how='left')

            # Calculate the total volume per order
            order_details_df['Total Volume cm3'] = order_details_df['Quantity'] * order_details_df['Volume cm3']


            # Read the Excel file into a pandas DataFrame
            distance_matrix_df = pd.read_excel('data/distance_matrix_C30.xlsx', sheet_name='Sheet1')

            # Convert the DataFrame to a NumPy array
            distance_matrix = distance_matrix_df.values

            # Retrieve the location of the main warehouse (W1)
            warehouse_location = warehouse_df.loc[warehouse_df['Ombor ID'] == 'W1', 'Location'].iloc[0]
            # Create a dictionary of locations including the warehouse and customer locations
            locations = {'W1': warehouse_location}
            locations.update(dict(zip(order_details_df['Customer ID'], order_details_df['Customer location'])))
            location_list = list(locations.keys())

            # Aggregate total demands per customer
            customer_demands = order_details_df.groupby('Customer ID')['Total Volume cm3'].sum()

            # Create a list for demands, starting with the depot
            demands = [0]  # starting with zero demand for the depot

            # Assuming the order of customers in 'customer_demands' matches the order of customer locations used in the distance matrix
            demands.extend(customer_demands.tolist())

            vehicle_capacities = vehicles_df['Vehicle Capacity cm3'].tolist()
            fuel_per_km = vehicles_df['fuel_per_km'].tolist()
            vehicle_ids = vehicles_df['Vehicle ID'].tolist()

            accessibility_dict = {}
            for index, row in customers_df.iterrows():
                # Convert customer ID from format 'C1' to integer 1
                customer_id = int(row['Customer ID'][1:])  # Assumes all IDs are formatted as 'C' followed by a number
                
                inaccessible_vehicles = []
                if pd.notna(row['not_accessible_by']):
                    # Split the string by commas and convert each vehicle ID from format 'V1' to integer
                    inaccessible_vehicles = [int(vehicle.strip()[1:]) for vehicle in row['not_accessible_by'].split(',') if vehicle.strip()]
                    # This strips spaces, removes the first character ('V'), and converts the remainder to an integer.

                accessibility_dict[customer_id] = inaccessible_vehicles
                
            ### ---------------------------------------- MODEL --------------------------------------
            # MOdel

            # Setup problem constants
            num_customers = len(customers_df)
            depot = 0
            customers = list(range(1, num_customers + 1))
            #vehicle_capacities = [20, 20, 20]  # Example capacities
            #demands = [0] + [random.randint(1, 5) for _ in range(num_customers)]

            # DEAP setup
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            toolbox = base.Toolbox()

            # Attribute generator
            def generate_customer():
                while True:
                    customer = random.choice(customers)
                    vehicle = random.choice(range(len(vehicle_capacities)))
                    if vehicle not in accessibility_dict.get(customer, []):
                        return customer

            toolbox.register("attribute", generate_customer)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, num_customers)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Evaluation function for CVRP with fuel cost minimization
            def evalCVRP(individual):
                total_fuel_cost = 0
                current_capacity = 0
                last_customer = depot
                vehicle_index = 0
                for customer in individual:
                    if current_capacity + demands[customer] > vehicle_capacities[vehicle_index]:
                        total_fuel_cost += distance_matrix[last_customer][depot] * fuel_per_km[vehicle_index]
                        current_capacity = 0
                        last_customer = depot
                        vehicle_index = (vehicle_index + 1) % len(vehicle_capacities)
                    total_fuel_cost += distance_matrix[last_customer][customer] * fuel_per_km[vehicle_index]
                    current_capacity += demands[customer]
                    last_customer = customer
                total_fuel_cost += distance_matrix[last_customer][depot] * fuel_per_km[vehicle_index]
                return (total_fuel_cost,)

            toolbox.register("evaluate", evalCVRP)

            # 2-opt optimization routine
            def apply_2opt(individual):
                improved = True
                while improved:
                    improved = False
                    for i in range(1, len(individual) - 2):
                        for j in range(i + 2, len(individual)):
                            if j - i == 1: continue  # Changes nothing, skip
                            new_route = individual[:i] + individual[i:j][::-1] + individual[j:]
                            if evalCVRP(new_route)[0] < evalCVRP(individual)[0]:
                                individual[:] = new_route
                                improved = True
                return individual

            # Repair function ensuring each customer is visited exactly once
            def repair_individual(individual):
                seen = set()
                repaired = []
                for customer in individual:
                    if customer not in seen:
                        repaired.append(customer)
                        seen.add(customer)
                missing = [customer for customer in customers if customer not in seen]
                random.shuffle(missing)
                repaired.extend(missing)
                return creator.Individual(repaired)

            # Custom mate and mutate functions that respect the constraints
            def custom_mate(ind1, ind2):
                ind1, ind2 = tools.cxTwoPoint(ind1, ind2)
                return repair_individual(ind1), repair_individual(ind2)

            def custom_mutate(ind):
                ind, = tools.mutShuffleIndexes(ind, indpb=0.05)
                ind = apply_2opt(ind)  # Apply 2-opt optimization directly in the mutate function
                return repair_individual(ind),  # Ensure repair is applied after mutation and 2-opt

            toolbox.register("mate", custom_mate)
            toolbox.register("mutate", custom_mutate)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Genetic Algorithm execution with eaMuCommaLambda
            def main():
                random.seed(42)
                population = toolbox.population(n=100)
                hof = tools.HallOfFame(1)
                stats = tools.Statistics(lambda ind: ind.fitness.values)
                stats.register("avg", np.mean)
                stats.register("min", np.min)
                stats.register("max", np.max)

                algorithms.eaMuCommaLambda(population, toolbox, mu=50, lambda_=100, cxpb=0.8, mutpb=0.1, ngen=10, stats=stats, halloffame=hof, verbose=True)
                return population, hof, stats

            #if __name__ == "__main__":
            #    main()
            ### ------------------------------------MODEL OUTCOME-------------------------------
            def print_routes(individual):
                route = []
                current_capacity = 0
                current_route = [depot]  # Start from the depot
                current_route_distance = 0
                total_distance = 0
                total_load = 0  # Initialize the total load counter
                total_fuel_cost = 0  # Initialize the total fuel cost counter
                vehicle_index = 0  # Initialize vehicle index

                outcome = {}  # Dictionary to store the outcome

                for customer in individual:
                    demand = demands[customer]
                    new_distance = distance_matrix[current_route[-1]][customer]

                    if current_capacity + demand > vehicle_capacities[vehicle_index]:  # Check capacity for the current vehicle
                        # Complete current route by returning to the depot
                        return_to_depot = distance_matrix[current_route[-1]][depot]
                        current_route_distance += return_to_depot
                        total_distance += current_route_distance
                        total_load += current_capacity

                        # Calculate fuel cost for the current route
                        current_fuel_cost = current_route_distance * fuel_per_km[vehicle_index]
                        total_fuel_cost += current_fuel_cost

                        # Output the current vehicle route
                        route_details = {
                            "Route": current_route + [depot],
                            "Distance": current_route_distance,
                            "Fuel Cost": current_fuel_cost,
                            "Load": current_capacity,
                            "Load Percentage": (current_capacity / vehicle_capacities[vehicle_index]) * 100
                        }
                        outcome[f"Vehicle {vehicle_index + 1}"] = route_details

                        # Prepare for the next vehicle
                        route.append(current_route)
                        vehicle_index = (vehicle_index + 1) % len(vehicle_capacities)
                        if len(vehicle_capacities) == 0:
                            raise ValueError("No vehicles available")
                        current_route = [depot, customer]
                        current_route_distance = distance_matrix[depot][customer]
                        current_capacity = demand
                    else:
                        current_route.append(customer)
                        current_route_distance += new_distance
                        current_capacity += demand

                # Finish the last route by returning to the depot
                if current_route:
                    return_to_depot = distance_matrix[current_route[-1]][depot]
                    current_route_distance += return_to_depot
                    total_distance += current_route_distance
                    total_load += current_capacity

                    # Calculate fuel cost for the last route
                    current_fuel_cost = current_route_distance * fuel_per_km[vehicle_index]
                    total_fuel_cost += current_fuel_cost

                    # Output the last vehicle route
                    current_route.append(depot)
                    route_details = {
                        "Route": current_route,
                        "Distance": current_route_distance,
                        "Fuel Cost": current_fuel_cost,
                        "Load": current_capacity,
                        "Load Percentage": (current_capacity / vehicle_capacities[vehicle_index]) * 100
                    }
                    outcome[f"Vehicle {vehicle_index + 1}"] = route_details

                # Output total distance and load for all vehicles
                outcome["Total"] = {
                    "Total Distance": total_distance,
                    "Total Load": total_load,
                    "Total Fuel Cost": total_fuel_cost
                }

                # print(outcome)
                return outcome


            # Main function call and evaluation
            if __name__ == "__main__":
                _, hof, _ = main()
                best_individual = hof[0]
                print("Best individual's route:")
                outcome = print_routes(best_individual)

            for vehicle in outcome.values():
                if 'Route' in vehicle:
                    vehicle['Route'] = [f'C{str(stop)}' for stop in vehicle['Route']]
                    
            ### --------------- Printing model outcome -------------------------------###
            # Define the CSS for the dataframe background color and text color
            def dataframe_to_styled_html(df):
                styled_df = df.style.hide(axis="index").set_table_styles(
                    [
                        {'selector': 'tbody tr', 'props': [('background-color', 'rgb(0, 43, 54)'), ('color', 'white')]},
                    ]
                ).to_html(classes='styled-table')
                return styled_df

            keys_list = list(outcome.keys())
            option_index = 0

            for key in keys_list:
                if key != "Total":
                    df_ = pd.DataFrame(outcome[key])
                    df1 = df_.drop(columns=['Distance', 'Fuel Cost', 'Load', 'Load Percentage'])
                    df = df1.transpose()
                    styled_html = dataframe_to_styled_html(df)
                    title_html = f'<h2 style="color:black; font-size:48px;">🚚 {options[option_index]}</h2>'
                    # Format values with appropriate precision
                    distance_km = round(df_["Distance"].values[0] / 1000)  # Round to nearest kilometer
                    load_percentage = df_["Load Percentage"].values[0]
                    fuel_cost_l = round(df_["Fuel Cost"].values[0] / 1000, 1)  # Round to 1 decimal place
                    title_html_t = f'<h2 style="color:black; font-size:18px;">Umumiy masofa : {distance_km:,} kilometr</h2>'
                    title_html_t1 = f'<h2 style="color:black; font-size:18px;">Umumiy yuklanganlik : {load_percentage:.1f} %</h2>'
                    title_html_t2 = f"<h2 style='color:black; font-size:18px;'>Sarflanadigan yoqilg'i : {fuel_cost_l:,} litr</h2>"

                    st.markdown(title_html, unsafe_allow_html=True)
                    st.markdown(styled_html, unsafe_allow_html=True)

                    # Display titles in a horizontal layout
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                        <div>{title_html_t}</div>
                        <div>{title_html_t1}</div>
                        <div>{title_html_t2}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    option_index = (option_index + 1) % len(options)  # Loop through options

            # Add some space before the total_df
            st.write("")
            st.write("")
            title_html_ = f'<h2 style="color:black; font-size:48px; text-align:center;">..UMUMIY..</h2>'
            st.markdown(title_html_, unsafe_allow_html=True)

            # Display the total_df
            total_df = pd.DataFrame(outcome['Total'], index=[0])
            total_df.columns = ['Umumiy masofa', 'Umumiy yuk', "Umumiy yonilgi"]

            # Format values with appropriate precision
            total_distance_km = round(total_df["Umumiy masofa"].values[0] / 1000)  # Round to nearest kilometer
            total_load_smkub = f"{total_df['Umumiy yuk'].values[0]:,.0f} smkub"  # Format with commas and add unit
            total_fuel_cost_l = round(total_df["Umumiy yonilgi"].values[0] / 1000, 1)  # Round to 1 decimal place

            total_html = f'<h2 style="color:white; font-size:20px; background-color:rgb(0, 43, 54); padding:10px;">Umumiy masofa : {total_distance_km:,} kilometr</h2>'
            st.markdown(total_html, unsafe_allow_html=True)

            total_html1 = f'<h2 style="color:white; font-size:20px; background-color:rgb(0, 43, 54); padding:10px;">Umumiy yuk : {total_load_smkub}</h2>'
            st.markdown(total_html1, unsafe_allow_html=True)

            total_html2 = f"<h2 style='color:white; font-size:20px; background-color:rgb(0, 43, 54); padding:10px;'>Umumiy yonilg'i : {total_fuel_cost_l:,} litr</h2>"
            st.markdown(total_html2, unsafe_allow_html=True)



            # Create two columns for the main dataframes
            # # Mapping of old keys to new keys
            # key_mapping = {
            #     'Route': 'Mort Depot',
            #     'Distance': 'Umumiy masofa',
            #     'Fuel Cost': "Umumiy sarflangan yoqilg'i",
            #     'Load Percentage': "Mashinaga to'lgan qismi"
            # }

            # # Function to update keys
            # def update_keys(d, mapping):
            #     new_dict = {}
            #     for k, v in d.items():
            #         if isinstance(v, dict):
            #             v = update_keys(v, mapping)
            #         new_dict[mapping.get(k, k)] = v
            #     return new_dict

            # # Updating the original dictionary
            # updated_data = update_keys(outcome, key_mapping)

            # key_list = list(updated_data.keys())
            
            # for key_ in key_list:
            #     final_dict = updated_data[key_]
            #     st.header(f"---------{key_}----------")
            #     for key, value in final_dict.items():
            #         st.write(f"{key} ----> {value}")

            #     st.write()  # Adds a newline between sections

    else:
        with st.sidebar:
            st.markdown('<p style="color: white;">Kamida 2 ta mashina tanlang</p>', unsafe_allow_html=True)
else:
    with st.sidebar:
        st.markdown('<p style="color: white;">Tanlangan mashinalarni tasdiqlashni unutmang !</p>', unsafe_allow_html=True)
        