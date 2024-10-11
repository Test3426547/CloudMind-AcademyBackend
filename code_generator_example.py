from fastapi_code_generator import CodeGenerator

# Define a sample OpenAPI specification
openapi_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "Sample API",
        "version": "1.0.0"
    },
    "paths": {
        "/items": {
            "get": {
                "summary": "List items",
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "name": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

# Generate code from the OpenAPI specification
generator = CodeGenerator(openapi_spec)
generated_code = generator.generate()

# Print the generated code
print(generated_code)

# You can also save the generated code to a file
with open("generated_api.py", "w") as f:
    f.write(generated_code)

print("Code generation complete. Check 'generated_api.py' for the result.")
