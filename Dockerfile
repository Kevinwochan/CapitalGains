<project name="example-suite" default="sign-jars">
    <!-- Register ant-contrib tasks -->
    <taskdef resource="net/sf/antcontrib/antlib.xml"/>

    <target name="sign-jars">
        <!-- Collect all JAR files under modules/**/dist/ -->
        <path id="jar.files">
            <fileset dir="modules">
                <include name="**/dist/*.jar"/>
            </fileset>
        </path>

        <!-- Iterate over each JAR file, signing each one -->
        <foreach param="jarfile">
            <path refid="jar.files"/>
            <sequential>
                <echo message="Signing @{jarfile}"/>
                <!-- Replace below with your jarsigner or custom signing implementation -->
                <!--
                <exec executable="jarsigner">
                    ...args for jarsigner/jsign...
                </exec>
                -->
            </sequential>
        </foreach>
    </target>
</project>


<target name="-post-jar">
  <property name="jarsigner" value="${env.JAVA_HOME}/bin/jarsigner"/>
  <property name="jsign.jar" value="/path/to/jsign-<version>.jar"/>
  <property name="signed.jar" value="${project.build.dir}/${ant.project.name}-signed.jar"/>
  <property name="aws.region" value="ap-southeast-2"/>
  <property name="aws.kms.key" value="arn:aws:kms:ap-southeast-2:123456789012:key/your-key-id"/>
  <property name="aws.credentials" value="access-key|secret-key|optional-session-token"/>
  <property name="certchain" value="/path/to/full-chain.pem"/>

  <exec executable="${jarsigner}">
    <arg line="-J-cp -J${jsign.jar}"/>
    <arg line="-providerClass net.jsign.jca.JsignJcaProvider"/>
    <arg line="-providerArg ${aws.region}"/>
    <arg line="-keystore NONE"/>
    <arg line="-storetype AWS"/>
    <arg line="-storepass ${aws.credentials}"/>
    <arg line="-keypass ${aws.credentials}"/>
    <arg line="-certchain ${certchain}"/>
    <arg value="${dist.jar}"/>
    <arg value="${aws.kms.key}"/>
  </exec>
</target>

<target name="sign-jar-with-aws-kms">
  <property name="jarsigner" value="${env.JAVA_HOME}/bin/jarsigner"/>
  <property name="jsign.jar" value="/path/to/jsign-<version>.jar"/>
  <property name="jar.to.sign" value="dist/my-module.jar"/>
  <property name="aws.region" value="ap-southeast-2"/>
  <property name="aws.kms.key" value="arn:aws:kms:ap-southeast-2:123456789012:key/your-key-id"/>
  <property name="aws.credentials" value="access-key|secret-key|optional-session-token"/>
  <property name="certchain" value="/path/to/full-chain.pem"/>

  <exec executable="${jarsigner}">
    <arg line="-J-cp -J${jsign.jar}"/>
    <arg line="-providerClass net.jsign.jca.JsignJcaProvider"/>
    <arg line="-providerArg ${aws.region}"/>
    <arg line="-keystore NONE"/>
    <arg line="-storetype AWS"/>
    <arg line="-storepass ${aws.credentials}"/>
    <arg line="-keypass ${aws.credentials}"/>
    <arg line="-certchain ${certchain}"/>
    <arg value="${jar.to.sign}"/>
    <arg value="${aws.kms.key}"/>
  </exec>
</target>


FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libsqlite3-dev \
    && pip install pipx \
    && rm -rf /var/lib/apt/lists/*

# Ensure pipx binaries are on the PATH
ENV PATH="$PATH:/root/.local/bin"

RUN pipx install uv

# Create virtual environment
RUN uv venv /app/.venv

# Activate virtual environment by setting PATH
ENV PATH="/app/.venv/bin:$PATH"

COPY ./app/requirements.txt requirements.txt

# Install packages into virtual environment
RUN uv pip install -r requirements.txt

COPY ./app /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
