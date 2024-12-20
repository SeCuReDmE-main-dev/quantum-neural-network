import unittest
import os
import subprocess

class TestIISApacheIntegration(unittest.TestCase):

    def test_iis_installation(self):
        result = subprocess.run(['powershell', 'Get-WindowsFeature', '-Name', 'Web-Server'], capture_output=True, text=True)
        self.assertIn('Installed', result.stdout)

    def test_iis_configuration(self):
        site_name = "NeuralNetworkCore"
        result = subprocess.run(['powershell', 'Get-Website', '-Name', site_name], capture_output=True, text=True)
        self.assertIn(site_name, result.stdout)

    def test_apache_ignite_configuration(self):
        ignite_path = "C:\\Apache\\Ignite"
        self.assertTrue(os.path.exists(ignite_path))
        self.assertTrue(os.path.exists(os.path.join(ignite_path, 'bin', 'ignite.bat')))

    def test_apache_mahout_configuration(self):
        mahout_path = "C:\\Apache\\Mahout"
        self.assertTrue(os.path.exists(mahout_path))
        self.assertTrue(os.path.exists(os.path.join(mahout_path, 'bin', 'mahout.bat')))

    def test_apache_avalanche_configuration(self):
        avalanche_path = "C:\\Apache\\Avalanche"
        self.assertTrue(os.path.exists(avalanche_path))
        self.assertTrue(os.path.exists(os.path.join(avalanche_path, 'bin', 'avalanche.bat')))

    def test_integration(self):
        site_path = "C:\\inetpub\\wwwroot\\NeuralNetworkCore"
        web_config_path = os.path.join(site_path, 'Web.config')
        self.assertTrue(os.path.exists(web_config_path))
        with open(web_config_path, 'r') as file:
            content = file.read()
            self.assertIn('ApacheIgnitePath', content)
            self.assertIn('ApacheMahoutPath', content)
            self.assertIn('ApacheAvalanchePath', content)

if __name__ == '__main__':
    unittest.main()
