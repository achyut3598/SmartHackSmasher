using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RoadSignDemo : MonoBehaviour
{
    public enum State { Initial, Driving, RoadSign, SpeedingUp, OnRamp, Explaining, Done };
    public State currentState;
    public GameObject textObjects;
    public float counter = 0f;

    public AudioSource hackingSound;
    public AudioSource stoppingSound;
    public AudioSource crashingSound;
    public AudioSource disappearSound;
    public AudioSource reappearSound;

    //Figure out how to display shit with the Speed Sign and Spedometer

    private RoadSignText textScript;

    private bool HasPlayedHackingSound = false;
    private bool HasPlayedStoppingSound = false;
    private bool HasPlayedCrashingSound = false;
    private bool HasPlayedDisappearSound = false;
    private bool HasPlayedSecondAppearSound = false;
    private bool HasPlayedAppearSound = false;

    // Start is called before the first frame update
    void Start()
    {
        currentState = State.Initial;
        textScript = textObjects.GetComponent<RoadSignText>();
        PauseGame();
    }

    // Update is called once per frame
    void Update()
    {
        counter += 1;
        switch (currentState)
        {
            case State.Initial:
                PauseGame();
                if (counter >= 300f)
                {
                    currentState = State.Driving;
                }
                break;
            case State.Driving:
                ResumeGame();
                textScript.changeToDrivingState();
                counter = 0f;
                break;
            case State.RoadSign:
                PauseGame();
                if (counter < 300f)
                {
                    textScript.changeToSignState();
                }
                if (counter > 300f)
                {
                    textScript.changeToSignExplainState();
                }
                if (counter > 900f)
                {
                    ResumeGame();
                    counter = 0f;
                    currentState = State.Driving;
                }
                break;
            case State.SpeedingUp:
                textScript.changeToSpeedUpState();
                break;
            case State.OnRamp:
                break;
            case State.Explaining:
                PauseGame();
                textScript.changeToExplainState();
                break;
            case State.Done:
                break;
        }
    }

    void PauseGame()
    {
        Time.timeScale = 0;
    }

    void ResumeGame()
    {
        Time.timeScale = 1;
    }
}
