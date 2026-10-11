# web/ - frontend memory (nested; loaded when Claude works inside web/)

## Commands
- Unit tests: `npm run test:unit -- <pattern>`; lint: `npm run lint`

## Conventions
- Function components with explicit prop types; no default exports.
- Data fetching goes through `src/client/` only; never call `fetch` from components.
- Tests live next to the component as `*.test.tsx`.
