using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NPCCar : MonoBehaviour
{
    public float speed = 20.0f;

    void Start()
    {

    }

    void Update()
    {
        transform.Translate(Vector3.forward * Time.deltaTime * speed);//Moves Forward based on Verticl Input
    }
}
