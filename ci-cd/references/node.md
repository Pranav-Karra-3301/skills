# Node.js / TypeScript CI/CD Configuration

## Pre-commit Setup with Husky v9

### Installation

```bash
# Install dependencies
npm install -D husky lint-staged @commitlint/cli @commitlint/config-conventional

# Initialize Husky (creates .husky directory)
npx husky init
```

### .husky/pre-commit

```bash
npx lint-staged
```

### .husky/commit-msg

```bash
npx --no -- commitlint --edit $1
```

### package.json lint-staged Configuration

```json
{
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml}": [
      "prettier --write"
    ]
  }
}
```

### commitlint.config.js

```javascript
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',     // New feature
        'fix',      // Bug fix
        'docs',     // Documentation
        'style',    // Formatting
        'refactor', // Code restructuring
        'perf',     // Performance
        'test',     // Tests
        'build',    // Build system
        'ci',       // CI configuration
        'chore',    // Maintenance
        'revert'    // Revert commit
      ]
    ],
    'subject-case': [2, 'always', 'lower-case'],
    'subject-empty': [2, 'never'],
    'type-empty': [2, 'never']
  }
};
```

## ESLint Security Configuration

### Installation

```bash
npm install -D eslint-plugin-security eslint-plugin-no-secrets
```

### eslint.config.js (Flat Config - ESLint 9+)

```javascript
import security from 'eslint-plugin-security';
import noSecrets from 'eslint-plugin-no-secrets';

export default [
  // ... other configs
  {
    plugins: {
      security,
      'no-secrets': noSecrets
    },
    rules: {
      // Security rules
      'security/detect-object-injection': 'warn',
      'security/detect-non-literal-regexp': 'warn',
      'security/detect-unsafe-regex': 'error',
      'security/detect-buffer-noassert': 'error',
      'security/detect-child-process': 'warn',
      'security/detect-disable-mustache-escape': 'error',
      'security/detect-eval-with-expression': 'error',
      'security/detect-no-csrf-before-method-override': 'error',
      'security/detect-non-literal-fs-filename': 'warn',
      'security/detect-non-literal-require': 'warn',
      'security/detect-possible-timing-attacks': 'warn',
      'security/detect-pseudoRandomBytes': 'error',
      'security/detect-new-buffer': 'error',

      // Secret detection
      'no-secrets/no-secrets': 'error'
    }
  }
];
```

### .eslintrc.js (Legacy Config - ESLint 8)

```javascript
module.exports = {
  plugins: ['security', 'no-secrets'],
  extends: ['plugin:security/recommended'],
  rules: {
    'no-secrets/no-secrets': 'error'
  }
};
```

## GitHub Actions CI Workflow

### .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18, 20, 22]

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run typecheck
        if: hashFiles('tsconfig.json') != ''

      - name: Test
        run: npm test

      - name: Build
        run: npm run build
        if: hashFiles('tsconfig.json') != '' || contains(github.event.repository.name, 'app')
```

### .github/workflows/ci.yml (pnpm variant)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup pnpm
        uses: pnpm/action-setup@v3
        with:
          version: 9

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Lint
        run: pnpm lint

      - name: Type check
        run: pnpm typecheck

      - name: Test
        run: pnpm test

      - name: Build
        run: pnpm build
```

## Full package.json Scripts Section

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint . --ext .js,.jsx,.ts,.tsx",
    "lint:fix": "eslint . --ext .js,.jsx,.ts,.tsx --fix",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "typecheck": "tsc --noEmit",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "prepare": "husky"
  },
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml,css,scss}": [
      "prettier --write"
    ]
  }
}
```

## Common Issues and Solutions

### Husky hooks not running

```bash
# Make hooks executable
chmod +x .husky/pre-commit
chmod +x .husky/commit-msg

# Reinstall hooks
npx husky install
```

### lint-staged failing on CI

lint-staged should only run locally. CI should run full linting:

```yaml
# CI runs full lint, not lint-staged
- name: Lint
  run: npm run lint
```

### ESLint security plugin false positives

```javascript
// Disable specific rule for a line
// eslint-disable-next-line security/detect-object-injection
const value = obj[userInput];

// Or configure in eslint config
rules: {
  'security/detect-object-injection': 'off' // if too noisy
}
```

### TypeScript strict mode conflicts

If enabling strict TypeScript checking causes too many errors:

```json
// tsconfig.json - gradual adoption
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,  // Enable one at a time
    "strictNullChecks": false
  }
}
```

## Recommended Dependencies

```bash
# Core
npm install -D husky lint-staged

# Commit linting
npm install -D @commitlint/cli @commitlint/config-conventional

# ESLint (if not present)
npm install -D eslint @eslint/js typescript-eslint

# Security plugins
npm install -D eslint-plugin-security eslint-plugin-no-secrets

# Formatting
npm install -D prettier eslint-config-prettier

# Testing (if not present)
npm install -D jest @types/jest ts-jest
```
