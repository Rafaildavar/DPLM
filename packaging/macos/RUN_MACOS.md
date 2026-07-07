# GestureBind macOS Bundle

## Run

1. Unzip the release artifact.
2. Open `GestureBind.app`.
3. If macOS blocks the first launch, use right click -> Open.
4. Allow camera access when macOS asks for permission.

## Notes

- The bundle is built for demo and contest validation, not notarized App Store
  distribution.
- The app uses a local SQLite fallback and stores runtime state under
  `~/.dplm/`.
- The bundle includes tracked model artifacts and configs from this repository.
- If Flet desktop runtime is not already cached on the target machine, the
  bundled Flet client archive is used. If that archive is missing or invalid,
  Flet may try to download its desktop client on first launch.

## Troubleshooting

If the app does not see the camera, check macOS Settings -> Privacy & Security
-> Camera and allow `GestureBind`.

If the app is blocked by Gatekeeper, open it with right click -> Open. For a
production build, the next CD step is Developer ID signing and notarization.
