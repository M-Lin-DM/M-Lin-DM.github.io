using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents.Sensors;
using MLAgents;
using System.Linq;
using System.IO;

public class Transporter_agent : Agent
{

    Rigidbody rb;
    public float RotationRate;
    public float speed;
    public int ID;

    string filename = "D:/ml-agents-master/ml-agents-master/ObservationData/test1.csv";

    private Transporter_Arena transporter_arena;//since this is the parent object it cant be linked in as a private var
    public GameObject Sphere;
    public GameObject Sphere1;
    [HideInInspector]
    public int item_carried = 0;//0,1,2
    public int has_passed = 0;//0,1,2
    public int has_received = 0;
    

    public override void Initialize()
    {
        base.Initialize();
        rb = GetComponent<Rigidbody>();
        transporter_arena = GetComponentInParent<Transporter_Arena>();

    }

    // // // Update is called once per frame
    // void FixedUpdate() //it doesnt seem possible to pull in data from the sensor components. specifically the raycast. unity may fix later
    // {
        
    //     using(var writer = new StreamWriter(filename, append: true))
    //     {
    //         // SensorComponent[] sensors = GetComponents<SensorComponent>();
    //         // RayPerceptionSensor sensor = gameObject.GetComponent<RayPerceptionSensor>();
    //         string AllSensorsObs = null;
    //         // foreach (var component in Sensorlist)
    //         // {
            
    //         float[] rowarray = sensor.M_Observations_get.Cast<float>().ToArray(); // cast it as an array of ints
    //         string sensor_string = string.Join(",", rowarray); //convert array to string separated by commas
    //         AllSensorsObs += "," + sensor_string;
            
    //         // }
    //         writer.WriteLine(AllSensorsObs);  //write each line to csv
    //     }
    // }

    
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.forward);
        sensor.AddOneHotObservation((int)item_carried, 3);
        sensor.AddOneHotObservation(ID, 6);


    }

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
        // Vector3 moveVector = transform.forward * forward_amount * speed * Time.fixedDeltaTime;
        // transform.position += moveVector;

        AddReward(-1f / maxStep);
        
    }

    private void OnCollisionEnter(Collision collision)
    {
        // Debug.Log("transformpos : " + transform.rotation.x.ToString("0.0 ") + transform.rotation.y.ToString("0.0 ") + transform.rotation.z.ToString("0.0 "));
        
         
        if ((collision.transform.CompareTag("spawn_box"))&(item_carried == 0))
        {
            if (item_carried == 0)
            {
                item_carried = 1;
                AddReward(1);
                GameObject myball = Instantiate<GameObject>(Sphere, transform.position + Vector3.up * 3f, Quaternion.Euler(0,0,0), transform);
            }  //   NOTE the upper ray cast offset (3) is with respect to the agent position. so must instantiate the sphere at Vector3.up * 3f
        }
        else if (collision.transform.CompareTag("sink_box"))
        {
            if (item_carried == 2)
            {
                item_carried = 0;
                AddReward(1);
                // Debug.Log("Attempting destroy 2");
                Destroy(gameObject.transform.GetChild(3).gameObject);
            }
        }        
        else if (collision.transform.CompareTag("agent"))
        {
            
            if ((item_carried == 1)&(collision.gameObject.GetComponent<Transporter_agent>().item_carried == 0))
            {
                
                AddReward(1);
                // Debug.Log("Attempting destroy 1"); //the issue is once the item carried changes in the first contacting agent, the second if statement cant be triggered in the contacted agent!
                Destroy(gameObject.transform.GetChild(3).gameObject);
                // StartCoroutine(wait_to_update());
                has_passed = 1; //placeholder indicator which will be corrected to 1 after other agents are updated
                // item_carried = 0;

            }
            else if ((item_carried == 0)&(collision.gameObject.GetComponent<Transporter_agent>().item_carried == 1))
            {
                
                AddReward(1);
                // Debug.Log("Attempting intant 1");
                GameObject myball = Instantiate<GameObject>(Sphere1, transform.position + Vector3.up * 3f, Quaternion.Euler(0,0,0), transform);
                // StartCoroutine(wait_to_update());
                has_received = 1;
                // item_carried = 2;
            }
        }  
    }
     
    IEnumerator wait_to_update()
    {
        //Print the time of when the function is first called.
        Debug.Log("Started Coroutine at timestamp : " + Time.time);

        //yield on a new YieldInstruction that waits for 5 seconds.
        yield return new WaitForSeconds(1f);

        //After we have waited 5 seconds print the time again.
        Debug.Log("Finished Coroutine at timestamp : " + Time.time);
    }

    public override float[] Heuristic()
    {
        var useraction = new float[2];
        // useraction should be all zero by default?
        if (Input.GetKey(KeyCode.UpArrow))
        {
            useraction[0] = 1f;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            useraction[1] = 1f; //clockwise rotation
        }
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            useraction[1] = 2f;//counterclock rot
        }
        return useraction;
    }

    public override void OnEpisodeBegin() //remember to reset any properties that need to be reset at episode start. any agent variables or states?
    {
        // transporter_arena.ResetArea();

        if (gameObject.transform.childCount > 3)
        {
          Destroy(gameObject.transform.GetChild(3).gameObject); 
          Destroy(gameObject.transform.GetChild(4).gameObject); //this appears to fix the duplicate balls.
        }
        item_carried = 0;//0,1,2
        has_passed = 0;//0,1,2
        has_received = 0;
        rb.angularVelocity = Vector3.zero;
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360), 0f));
        transform.position = new Vector3(Random.Range(-20, 20), 0.5f, Random.Range(-20, 20));  
        
        
    }
}
