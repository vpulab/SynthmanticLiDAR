#!/usr/bin/env python

# ==============================================================================
#
# This script is used to generate a dataset using carla, for visual odometry
# 
#
# ==============================================================================


# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import queue
import pygame
import numpy as np
import math
import argparse
import yaml

########################### Change to the location of PythonAPI in your computer ####################
try:
    sys.path.append(glob.glob('D:/carla_project/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def save_scan(scan,environment_ready,counter):
    max_scans = 4500
    if counter >= max_scans:
        exit()

    if environment_ready == True:
        scan.save_to_disk('_out/{}.ply' % scan.frame) # Save the scan
        counter = counter+1
    #Add to poses list


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []    

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

# Function to change rotations in CARLA from left-handed to right-handed reference frame
def rotation_carla(rotation):
    cr = math.cos(math.radians(rotation.roll))
    sr = math.sin(math.radians(rotation.roll))
    cp = math.cos(math.radians(rotation.pitch))
    sp = math.sin(math.radians(rotation.pitch))
    cy = math.cos(math.radians(rotation.yaw))
    sy = math.sin(math.radians(rotation.yaw))
    return np.array([[cy*cp, -cy*sp*sr+sy*cr, -cy*sp*cr-sy*sr],[-sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],[sp, cp*sr, cp*cr]])

# Function to change translations in CARLA from left-handed to right-handed reference frame
def translation_carla(location):
    if isinstance(location, np.ndarray):
        return location*(np.array([[1],[1],[1]]))
    else:
        return np.array([location.x, location.y, location.z])

def main(config):
    actor_list = []
    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.
    counter = 0
    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)


        pygame.init() # To be able to exit
        
        
        clock = pygame.time.Clock()
        # Once we have a client we can retrieve the world that is currently
        # running.
        print(client.get_available_maps())

        #world = client.load_world("/Game/Carla/Maps/"+config['map'])

        #exit()
        world = client.get_world()
        
        
        #############################################################################
        # Set up synchronous mode
        #############################################################################

        settings = world.get_settings()
        fps = 10
        settings.fixed_delta_seconds = (1.0 / fps) if fps > 0.0 else 0.0
        settings.synchronous_mode = True
        ready = False
        synchronous_master = True
        world.apply_settings(settings)

        #client.reload_world(False)

        respawn = False
        hybrid = False
        seed = config['seed']
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        tm_port = traffic_manager.get_port()
        if seed is not None:
            traffic_manager.set_random_device_seed(seed)

        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        
        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = random.choice(blueprint_library.filter('vehicle.mercedes.coupe'))

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        camera_spawn = 10
        start_pose = world.get_map().get_spawn_points()[camera_spawn]
        waypoint = world.get_map().get_waypoint(start_pose.location)

        camera_spawn = 20


        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, start_pose)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True,tm_port)
        

        #######################################################################################
        # Spawn Vehicles and People
        #######################################################################################
        
        batch = []

        filterv = 'vehicle'
        generationv = 'All'
        blueprints = get_actor_blueprints(world, filterv, generationv)
        safe = False

        if safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            #blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            #blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            #blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        car_blueprints = blueprints
        motorbike_blueprints = blueprints
        bike_blueprints = blueprints
        truck_blueprints = blueprints
        #filter cars (remove motorbikes,bikes and trucks)
        car_blueprints = [x for x in car_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('microlino')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('cybertruck')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('sprinter')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('firetruck')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('ambulance')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('carlacola')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('truck')]

        motorbike_blueprints = [x for x in motorbike_blueprints if x.id.endswith('low_rider') | x.id.endswith('ninja') | x.id.endswith('zx125') | x.id.endswith('yzf')]
        bike_blueprints = [x for x in bike_blueprints if x.id.endswith('omafiets') | x.id.endswith('century') | x.id.endswith('crossbike')]
        truck_blueprints = [x for x in truck_blueprints if x.id.endswith('firetruck') | x.id.endswith('carlacola') | x.id.endswith('ambulance') | x.id.endswith('truck')]


        #blueprints = [blueprint_library.filter('kitti')]
        spawn_points = world.get_map().get_spawn_points()[:camera_spawn] + world.get_map().get_spawn_points()[camera_spawn:]

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        number_of_vehicles = config['vehicle_amounts'][0]['cars'] +config['vehicle_amounts'][2]['bikes'] + config['vehicle_amounts'][3]['trucks']+ config['vehicle_amounts'][1]['motorbikes']
        hero = False
        sensor_placed = False
 
        vehicle_amounts = {'car':(config['vehicle_amounts'][0]['cars'],car_blueprints),'bike':(config['vehicle_amounts'][2]['bikes'],bike_blueprints),'truck':(config['vehicle_amounts'][3]['trucks'],truck_blueprints),'motorbike':(config['vehicle_amounts'][1]['motorbikes'],motorbike_blueprints)} #must add to spawn points
        

        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break

            veh_type = random.choice(list(vehicle_amounts.keys()))

            if vehicle_amounts[veh_type][0] <= 0:
                vehicle_amounts.pop(veh_type)
                continue
            blueprint = random.choice(vehicle_amounts[veh_type][1])

            vehicle_amounts[veh_type] = (vehicle_amounts[veh_type][0]-1,vehicle_amounts[veh_type][1])

            if vehicle_amounts[veh_type][0] <= 0:
                vehicle_amounts.pop(veh_type)

            print(blueprint.id)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            if sensor_placed:# spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
            else:
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                print("Spawn Error")
            else:
                actor_list.append(response.actor_id)

        used_spawn = []

        for i in range(0, 15):
            #transform = world.get_map().get_spawn_points()[i+1+20]

            bp = random.choice(blueprint_library.filter('walker'))
            controller = blueprint_library.find('controller.ai.walker')
            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            trans = carla.Transform()
            got_location = world.get_random_location_from_navigation()
            while got_location in used_spawn:
                got_location = world.get_random_location_from_navigation()

            used_spawn.append(got_location)

            
            trans.location = got_location
            trans.location.z+=1
            npc = world.try_spawn_actor(bp, trans)
            world.tick()

            actor_controllers = []
            
            if npc is not None:
                controller_sp = world.spawn_actor(controller,carla.Transform(),npc)
                world.tick()
                controller_sp.start()
                controller_sp.go_to_location(world.get_random_location_from_navigation())
                actor_controllers.append(controller_sp)
                actor_controllers.append(npc)
                print('created %s' % npc.type_id)

        traffic_manager.global_percentage_speed_difference(40.0)


        ####################################################################################
        # Sensor initialization
        ####################################################################################
        sensor_list = []
        #Common parameters
        image_size_x = 1920
        image_size_y = 1080
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.63))#, carla.Rotation(pitch=-15))

        # #### RGB Cameras ####
        # # RGB Camera 1
        # # Find blueprint
        # camera_bp = blueprint_library.find('sensor.camera.rgb')
        # #Configure camera parameters
        # camera_bp.set_attribute('fov',str(120)) #In cm
        # camera_bp.set_attribute('image_size_x',str(image_size_x))
        # camera_bp.set_attribute('image_size_y',str(image_size_y))

        # #Spawn the camera sensor
        # camera_rgb_1 = world.spawn_actor(
        #     camera_bp,
        #     camera_transform,
        #     attach_to=vehicle)
        # sensor_list.append(camera_rgb_1)

        ### LiDAR ###

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        semantic_s_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
        semantic_s_bp.set_attribute('upper_fov',str(2))
        semantic_s_bp.set_attribute('lower_fov',str(-25))
        semantic_s_bp.set_attribute('range', '50.0')
        semantic_s_bp.set_attribute('channels', '64')
        #semantic_s_bp.set_attribute('points_per_second',str(10*64*360/0.8))
        semantic_s_bp.set_attribute('points_per_second',str(5*64*360/0.08))
        semantic_s = world.spawn_actor(semantic_s_bp,  carla.Transform(carla.Location(z=1.63)), attach_to=actor_list[0])
        sensor_list.append(semantic_s)

        dataset_path = 'D:/semlidar/dataset/sequences/70/'

        first_frame = True
        with open(dataset_path+"poses.txt", 'w') as posfile:
             posfile.write("## {} {} {} {} {} {}".format("roll","pitch","yaw","x","y","z"))



        ##############################################################################################
        # Create a synchronous mode context.
        ##############################################################################################
        with CarlaSyncMode(world, semantic_s, fps=10) as sync_mode:
            counter = 0
            while True:
                if should_quit():
                    return
                clock.tick()

                
                # Advance the simulation and wait for the data.
                snapshot, semantic_scan = sync_mode.tick(timeout=10.0) #Ajusta timeout si el pc es muy lento
                if first_frame:
                    #initial_camera_position = vehicle.get_location() + camera_transform.location
                    initial_camera_rotation = vehicle.get_transform().rotation
                    initial_camera_position = actor_list[0].get_location() + carla.Location(z=1.63)
                    print("Initial camera position (absolute): {}".format(initial_camera_position))
                    first_frame = False

                # Choose the next waypoint and update the car location.
                #waypoint = random.choice(waypoint.next(1.5))
                #vehicle.set_transform(waypoint.transform)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Lets save position and rotation (roll pitch yaw) as global values and will process them later...
                #rotation_v = np.matmul(rotation_carla(initial_camera_rotation).T,rotation_carla(lidar_scan.transform.rotation))
                #translation_v = np.array(translation_carla(vehicle.get_location() + carla.Location(z=1.63)) - translation_carla(initial_camera_position))
                #print(translation_v)

                # Save the scans
                counter+=1
                semantic_scan.save_to_disk(dataset_path+'/scan/%06d.ply' % semantic_scan.frame) # Save the scan

                
                #Save poses
                with open(dataset_path+"poses.txt", 'a') as posfile:
                    posfile.write("{} {} {} {} {} {}".format(semantic_scan.transform.rotation.roll,semantic_scan.transform.rotation.pitch,semantic_scan.transform.rotation.yaw,semantic_scan.transform.location.x,semantic_scan.transform.location.y,semantic_scan.transform.location.z))
                    #posfile.write(" ".join(map(str,[r for r in rotation_v[0]]))+" "+str(translation_v[1])+" ")
                    #posfile.write(" ".join(map(str,[r for r in rotation_v[1]]))+" "+str(translation_v[0])+" ")
                    #posfile.write(" ".join(map(str,[r for r in rotation_v[2]]))+" "+str(translation_v[2])+" ")
                    posfile.write("\n")

                if counter >= config['sampling_steps']:
                    return
    finally:
        if synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(actor_controllers), 2):
            actor_controllers[i].stop()

        for i in sensor_list:
            i.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_controllers[::2]])

        time.sleep(0.5)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CARLA Visual Odometry Dataset Generator')
    parser.add_argument('--config_file','-c', help='Config file',required=True)
    parser.add_argument('--output_folder','-o', help='Output folder',required=True)
    
    args = parser.parse_args()

    with(open(args.config_file)) as f:
        config = yaml.safe_load(f)

    main(config)
