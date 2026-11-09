import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from fabsim.cli import main

    raise SystemExit(main())
