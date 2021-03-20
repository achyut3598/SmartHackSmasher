using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StopSignDemonstration : MonoBehaviour
{
    public enum State { Initial, Driving, Slowing, Stopping, Turning, Crashing, Explaining, Done };
    public State currentState;
    public GameObject textObjects;
    public float counter = 0f;

    public AudioSource hackingSound;
    public AudioSource stoppingSound;
    public AudioSource crashingSound;
    public AudioSource disappearSound;
    public AudioSource reappearSound;



    private StopSignText textScript;

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
        textScript = textObjects.GetComponent<StopSignText>();
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
            case State.Slowing:
                PauseGame();
                textScript.changeToSlowingState();
                if (counter > 300f)
                {
                    ResumeGame();
                    stoppingSound.Play();
                    counter = 0f;
                    currentState = State.Driving;
                }
                break;
            case State.Stopping:
                break;
            case State.Turning:
                PauseGame();
                textScript.changeToTurningState();
                if (counter > 300f)
                {
                    ResumeGame();
                    counter = 0f;
                    currentState = State.Driving;
                }
                break;
            case State.Crashing:
                if (!HasPlayedCrashingSound)
                {
                    crashingSound.Play();
                    HasPlayedCrashingSound = true;
                }
                PauseGame();
                textScript.changeToCrashState();
                if (counter > 300f)
                {
                    ResumeGame();
                    counter = 0f;
                    currentState = State.Driving;
                }
                break;
            case State.Explaining:
                PauseGame();
                textScript.changeToExplanationState();
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
