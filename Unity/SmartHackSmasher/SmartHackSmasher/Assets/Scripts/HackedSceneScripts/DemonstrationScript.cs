using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DemonstrationScript : MonoBehaviour
{
    public enum State { Initial, Driving, Slowing, Hacking, Parking, Explaining, Crashing, Done};
    public State currentState;
    public GameObject textObjects;
    public GameObject normalParkedCars;
    public GameObject hackedParkedCars;
    public float counter = 0f;

    public AudioSource hackingSound;
    public AudioSource stoppingSound;
    public AudioSource crashingSound;
    public AudioSource disappearSound;
    public AudioSource reappearSound;

    

    private TextScript textScript;

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
        textScript = textObjects.GetComponent<TextScript>();
        PauseGame();
    }

    // Update is called once per frame
    void Update()
    {
        counter += 1;
        switch (currentState)
        {
            case State.Initial:
                //Progress the gameState after 5 Seconds
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
                if (!HasPlayedStoppingSound)
                {
                    stoppingSound.Play();
                    HasPlayedStoppingSound = true;
                }
                break;
            case State.Hacking:
                if (counter > 30f)
                {
                    if (!HasPlayedHackingSound)
                    {
                        hackingSound.Play();
                        HasPlayedHackingSound = true;
                    }
                    textScript.changeToHackingState();
                    PauseGame();
                    //Progress the gameState after 5 Seconds
                }
                if (counter >= 420f)
                {
                    currentState = State.Driving;
                }
                break;
            case State.Parking:
                if (!HasPlayedDisappearSound)
                {
                    disappearSound.Play();
                    HasPlayedDisappearSound = true;
                }
                normalParkedCars.transform.localScale = new Vector3(0, 0, 0);
                hackedParkedCars.transform.localScale = new Vector3(0, 0, 0);
                if (counter > 60f)
                {
                    textScript.changeToParkingState();

                    //Progress the gameState after 5 Seconds
                    PauseGame();
                }   
                if (counter >= 360f)
                {
                    currentState = State.Driving;
                }
                break;
            case State.Explaining:
                if (!HasPlayedAppearSound)
                {
                    reappearSound.Play();
                    HasPlayedAppearSound = true;
                }
                normalParkedCars.transform.localScale = new Vector3(1, 1, 1);
                textScript.changeToExplanationState();
                //Progress the gameState after 5 Seconds
                PauseGame();
                if (counter >= 600f)
                {
                    currentState = State.Driving;
                }
                break;
            case State.Crashing:
                if (!HasPlayedCrashingSound)
                {
                    crashingSound.Play();
                    HasPlayedCrashingSound = true;
                }
                if (counter > 60f)
                {
                    textScript.changeToCrashState();
                    if (!HasPlayedSecondAppearSound)
                    {
                        reappearSound.Play();
                        HasPlayedSecondAppearSound = true;
                    }
                    hackedParkedCars.transform.localScale = new Vector3(1, 1, 1);
                    //Progress the gameState after 5 Seconds
                    PauseGame();
                }
                if (counter >= 420f)
                {
                    currentState = State.Done;
                }
                break;
            case State.Done:
                ResumeGame();
                textScript.changeToDrivingState();
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
