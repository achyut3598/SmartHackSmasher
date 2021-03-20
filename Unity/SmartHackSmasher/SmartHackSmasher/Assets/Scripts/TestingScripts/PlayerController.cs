using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public float speed = 20.0f;
    private float turnSpeed = 45.0f;
    private float horizontalInput;
    private float forwardInput;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        forwardInput = Input.GetAxis("Vertical");
        horizontalInput = Input.GetAxis("Horizontal");
        //Move the Vehicle Forward
        //transform.Translate(0, 0, 1);
        //transform.Translate(Vector3.forward * Time.deltaTime * 20);
        transform.Translate(Vector3.forward * Time.deltaTime * speed * forwardInput);//Moves Forward based on Verticl Input
        //transform.Translate(Vector3.right * Time.deltaTime * turnSpeed * horizontalInput);
        transform.Rotate(Vector3.up, Time.deltaTime * turnSpeed * horizontalInput);//Rotates based on Horizontal Input
    }
}
