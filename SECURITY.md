# Security Policy

## Reporting Security Vulnerabilities

We take security seriously. If you discover a security vulnerability, please email **abdelkaderboukart@gmail.com** instead of using the issue tracker.

### Responsible Disclosure

When reporting a security vulnerability, please:

1. **Do not** open a public GitHub issue
2. **Do not** share vulnerability details publicly
3. Email detailed information about the vulnerability to our security team
4. Allow us reasonable time to address the issue before public disclosure

### What to Include

- Vulnerability type and description
- Affected code or component
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

## Security Best Practices

Our project follows these security practices:

- **Dependency Scanning**: Automated scans for known vulnerabilities (Safety, pip-audit)
- **Code Scanning**: Static analysis with CodeQL and Semgrep
- **Container Scanning**: Image scanning with Trivy and Grype
- **Secret Scanning**: Automated detection of credentials using TruffleHog
- **SBOM Generation**: Software Bill of Materials for supply chain security
- **Dependabot Updates**: Automated dependency updates for security patches

## Security Scanning Tools

The project uses multiple security tools:

- **CodeQL**: Semantic code analysis for detecting vulnerabilities
- **Semgrep**: Pattern-based security scanning
- **Trivy**: Container image vulnerability scanning
- **Grype**: Vulnerability scanner for container images
- **Bandit**: Security issue detection in Python code
- **TruffleHog**: Secret and credential detection
- **Safety**: Python dependency vulnerability scanning
- **pip-audit**: Audit Python packages for known vulnerabilities

## Supported Versions

| Version | Supported       |
| ------- | --------------- |
| 3.13    | ✓ Active        |
| 3.12    | ✓ Active        |
| 3.11    | ✓ Active        |
| < 3.11  | ✗ Not Supported |

## Acknowledgments

We appreciate responsible security researchers who help improve our security posture.
