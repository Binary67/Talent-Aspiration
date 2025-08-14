Project Working Guidelines

## File and Folder Organization
- Work within the existing project structure
- Use Pascal Case for all new folders and files (e.g., UserService.js, DataModels/, ApiControllers/)
- Place all unit tests in the UnitTests/ folder with corresponding Pascal Case naming (e.g., UserServiceTests.js for UserService.js)
- Maintain parallel folder structure in UnitTests/ that mirrors the source code organization

## Code Development Practices
- Apply DRY (Don't Repeat Yourself) principle rigorously
- Extract common functionality into reusable functions, classes, or modules
- Create utility functions for operations used more than twice
- Use inheritance, composition, or higher-order functions to eliminate code duplication
- When similar logic appears in multiple places, refactor into a shared component

## Validation Requirements
- Run lint checks before finalizing any code changes
- Fix all linting errors and warnings
- Write comprehensive unit tests for all new functions and classes
- Ensure unit tests cover edge cases, error scenarios, and happy paths
- Place tests in UnitTests/ folder maintaining the source code structure
- Run all existing tests to ensure no regressions

## Research and Problem Solving
- When encountering unfamiliar libraries or APIs, search the web for documentation and examples
- If a library method is unclear, examine the library's source code package repository
