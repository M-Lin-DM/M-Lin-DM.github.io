--- 
title: "The impact of rudimentary self-awareness on specialization in a cooperative multi-agent RL setting"
date: 2020-04-17
layout: "single"
permalink: /Transporters/
tagline: ""
header:
  overlay_image: /images/transporters/capture.JPG
mathjax: "true"
---
# In Brief
- 3
- dsf
- sdf
- sdf

# Intro and Concepts explored by this project

## Learned behavioral modes directly linked to binary observations

In single-agent deep reinforcement learning scenarios, it is often beneficial to provide the agent with information on their internal state. In problems where the agent must learn to control the movement of its body in order to walk for example, the angles and relative positions of all its joints are given as an observation. A neural network model must then learn a mapping between this bodily configuration and an appropriate action effecting movment. 

Observations based on "self-awareness" can be much simpler in fact. Agents trained with an observation vector including a binary state such as "I'm hungry = 0/1" can exhibit dramatic behavioral shifts when this state changes. For example, the agent may ignore a food source until im-hungry switches from 0 to 1, and it will then begin approaching it. In a previous version of the project described here, I found that providing agents information on the object they are currently carrying can cause them to exhibit completely different *behvioral modes* when this object changes. The information, called "item carried," was represented as a one-hot encoding of three mutually exclusive states: {empty-handed, carrying item 1, carrying item 2}. When agents began carrying item 2, for example, they begin moving in the +x direction (Fig ).

This begs the question of what would happen if agents were trained using observations which include a one-hot encoded *ID number* (which is to remain fixed during inference.) **Through training, will individuals evolve different behavioral sub-types as a result of perceiving this "ID state" in the same way that a single agent learns distinct behavioral modes?**

### How can a single neural network input make such a significant behavioral difference? 
The inclusion of a one-hot encoded ID implicitly allows agents **partially unshared** neural network parameters in the first dense layer. An agent has exclusive access to a subset of weights in this layer (fig below). In the single-agent case, the influence of a single binary state is evidently enough to induce a different behavioral mode. This suggests that including one-hot encoded IDs may be sufficient to allow emergent behavioral differentiation through the process of training. In this project I assess the impact of providing each agent their own one-hot encoded ID number as part of their observation (or state). 
![](/images/transporters/onehotNN.JPG)
*Fig. *
## Obligatory vs. Emergent Cooperation
In multi-agent reinforcement learning and agent-based modeling, it is important to make a distiction between the (more interesting) "emergent" cooperation and what I would call "obligatory" or "prescribed" cooperation. This project uses the latter. I define emergent cooperation as cooperative behavior (agents working together to do a task) which has not been directly incentivized and is not absolutely required to perform well on a task. Obligatory cooperation on the other hand has been strongly incentivized, possibly using rewards that reinforce the cooperative act directly. My project directly reinforces cooperative acts and even makes such acts necessary in order to complete the task. 

## "Functional" vs "Intrinsic" Specialization

# Reinforcement learning problem to be solved
I train agents to acheive a group-level objective: transporting items across a space as fast as possible. The game is designed so that it **requires** cooperation to complete the task. Specifically, agents can retrieve green spheres (item 1) from the source box, but they can't deposit them in the sink box themselves. Instead they must pass it to an empty-handed neighbor, who can then deposit it in the sink box. That is, only those who have received item 1 from a neighbor can deposit it. This is achieved by automatically converting the item 1 to item 2 when it is received.

To obtain item 1, pass it, and deposit item 2, agents must collide. The item transfer occurs at the same frame the collision occured. A reward of +1 is triggered by each of the three events and by no other collision types. In the case of passing, the reward is given to both the passer and receiver at the time of collison. Therefore, cooperation is not only required but incentivized directly. This contrasts with several works in multi-agent deep RL which achieve emergent cooperative behavior (ie without direct incentivization or requirement.) My study is directed more towards emergent diversity in a necessarily cooperative task.
![](/images/transporters/transfer_on_contact.JPG)
*Fig. Agents (yellow cylinders) obtain and transfer items (spheres) at the moment of collision. An empty handed agent (item =0) must first retrieve item 1 from the source box (green). It can then transfer it to an empty-handed agent in step 2. The item will then self-convert to item 2 (red sphere). Finally the agent can deposit item 2 in the sink box (red cylinder)*

### Observations
Agents receive the one-hot encoded item type $$\{0,1,2\}$$ that they are carrying $$V_{item-carried} \in \{0,1\}^{3}$$. This is critical to include as it allows the agent to adopt different behavioral when their item_carried changes. In practice, sudden shifts in behavior are common when this part of the observation vector changes (see video).

In most multi-agent settings it is necessary to let agents perceive information about each other. I achieve this by physically manifesting the agents "item_carried" state as a sphere that moves in sync with the agent at all times. I then allow other agents to perceive the state of their neighbors via a dual ray cast sensor. The upper tier ray cast detects only the arena walls and the item being carried (tagged sphere1/sphere2) above other agents (a total of 3 objects). The lower ray cast detects the source and sink boxes and other agents (3 objects). Each ray cast sensor comprises a vector $$V_{raycast-lower} \in \mathbb{R}^{35}$$, $$V_{raycast-upper} \in \mathbb{R}^{35}$$
![](/images/transporters/raycast2level_2.JPG)
![](/images/transporters/raycast2level.JPG)
*Fig. Agents use two-layer raycast percpetions in addition to their one-hot encoded ID and forward-facing vector. The top layer is able to detect the item carried by neighbors and the walls. The bottom layer detects the source and sink boxes and other agents.*

In addtion to ray casts, each agent receives its one-hot encoded ID: $$V_{ID} \in \{0,1\}^6$$ and its forward-facing vector normalized to length 1: $$V_{forward} \in \mathbb{R}^3$$. The forward facing vector acts like a compass and aids in agents orienting themselves. It may speed up learning of good policies; however it leads to inflexible behavior when the environment is changed (see Results).

Ultimately all observations are concatenated and fed to a neural network with 4 densely connected layers. 


```csharp
    {
        sensor.AddObservation(transform.forward);
        sensor.AddOneHotObservation((int)item_carried, 3);
        sensor.AddOneHotObservation(ID, 6); //"ID" can be changed to a specific value e.g 1 to force all agents to act like agent 1.
    }
```

### Actions
Agents use a multi-discrete action space with 2 action branches. This allows them to select one action from each branch at the same time step and then do both.

branch 0: $$\{0, 1\}$$ = {no movement, move forward} (by some increment)

branch 1: $$\{0, 1, 2\}$$ = {no rotation, rotate counterclockwise, rotate clockwise} (by some increment)

```csharp
    public override void OnActionReceived(float[] vectorAction)
    {
        var forward_amount = vectorAction[0];
        var rotationAxis = (int)vectorAction[1]; 
        var rot_dir = Vector3.zero;
        

        switch (rotationAxis)
        {
            case 1:
            rot_dir = Vector3.up*1f;
            break;

            case 2:
            rot_dir = Vector3.up*-1f;
            break;
        }

        transform.Rotate(rot_dir, RotationRate * Time.fixedDeltaTime, Space.Self);
        rb.MovePosition(transform.position + transform.forward * forward_amount * speed * Time.fixedDeltaTime);

        AddReward(-1f / maxStep);   
    }

```


# Results

## Scaling the game: Including more agents increases performance

## Changing the environment leads to task failure


## Partially shared (or rather, *unshared*) parameters led to increased specialization
![](/images/transporters/tablefofe.JPG)
*caption*

![](/images/transporters/SD.JPG)
![](/images/transporters/SD0.JPG)
![](/images/transporters/SD1.JPG)
*caption*


## Group performance did not benefit from increased specialization
sfdsdfsf
sdfs
![](/images/transporters/CumulativeR.JPG)
*caption*

## The three behavioral modes shown by the observation vectors' data manifold
sdfsdfsf
sdf
![](/images/transporters/agent0_item.JPG)
*caption*
sdfsdf
sdf

![](/images/transporters/agent0xcomp.JPG)
*caption*
